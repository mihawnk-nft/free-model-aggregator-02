// free-model-aggregator-02: one server, two client wires, one real free provider.
//
// The entry serves the OpenAI Chat Completions API and the Anthropic Messages API
// (JSON and server-sent events) and answers every prompt with a REAL completion
// from one of the fourteen free OpenRouter models, dialled through the grader's
// gateway on Chat Completions. Every model is dialled on the one upstream wire:
// an Anthropic-shaped client request is translated into a Chat Completions body
// first and the gateway's answer translated back.
//
// Routing, in one place. Candidates are every free model meeting the request's
// requirements, ranked by the brief's rating_milli (quality is 60% of the
// score); ties go to the model with the fewest calls so far, which spreads the
// load over the models' own rate limits instead of draining one. A model that
// has refused is set aside: it is tried only after every model that has not
// refused, because a refusal means it is limited right now and dialling it
// first on the next prompt wastes a gateway call (the score's tie-break counts
// calls, and the trial's key has a daily ceiling) and usually refuses again.
// A set-aside model is not dead — it is still tried when every clean candidate
// refuses, and it becomes clean again the moment it answers. Three refusals
// in a row spend a model for the trial. The trail of refusals is reported on
// the response as x-makefun-upstream-failures, the only way the failover
// column can see a refusal the entry retried and then served. The catalog is
// exactly the fourteen free ids the brief names: register that same set and
// answer GET /v1/models with it on both auth headers.
//
// The server is deterministic: every route is a function of the requests and
// the gateway's answers seen so far, never of the wall clock, so a live pass
// and its replay make the same calls. The catalog is built once, printed byte
// for byte by `--list` and posted in the register body.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take, in milliseconds. Real free models need seconds, not the 5 s a simulated provider did. */
const PROVIDER_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Consecutive refusals after which a model is treated as spent for the trial. */
const DEAD_AFTER_FAILURES = 3;
/** The most refusal rows the failures header carries. */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** What this trial has learned about one model: calls made, refusals in a row, spent or not. */
interface ModelState {
  calls: number;
  consecutiveFailures: number;
  dead: boolean;
}

const states = new Map<string, ModelState>();

function stateOf(modelId: string): ModelState {
  let state = states.get(modelId);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, dead: false };
    states.set(modelId, state);
  }
  return state;
}

/** One refusal on the trail a served response reports. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string;
}

/**
 * Book one refusal against a model: its run of refusals grows, three in a row
 * spends it, and the refusal goes on the trail the served response reports.
 * Used for a gateway refusal and for a fetch that never reached the gateway.
 */
function recordRefusal(model: ModelRow, failures: UpstreamFailure[], status: number, code: string): void {
  const state = stateOf(model.id);
  state.consecutiveFailures += 1;
  if (state.consecutiveFailures >= DEAD_AFTER_FAILURES) state.dead = true;
  failures.push({ status, code, model: model.id });
}

/**
 * The free models that can serve a request, best first.
 *
 * Order: models that have not refused come before models that have (a refusal
 * means the model is limited right now, so trying it first on the next prompt
 * spends a gateway call on an answer it will probably refuse again); inside
 * each group the brief's rating, then the model with the fewest calls so far
 * (load spreading), then the brief's order (Array.sort is stable, so equal
 * keys keep it). Spent models are left out. A set-aside model is still tried
 * when every clean candidate refuses, and answering clears its run of
 * refusals, so it is clean again for the next prompt.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  // Read each model's state once: the sort below must see one snapshot, and
  // a comparator that re-reads the map on every comparison hides that.
  const ranked = freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)) && !stateOf(model.id).dead)
    .map((model) => {
      const state = stateOf(model.id);
      return { model, setAside: state.consecutiveFailures > 0, rating: model.rating_milli, calls: state.calls };
    });
  ranked.sort((a, b) => {
    if (a.setAside !== b.setAside) return a.setAside ? 1 : -1;
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.calls !== b.calls) return a.calls - b.calls;
    return 0;
  });
  return ranked.map((entry) => entry.model);
}

/** The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Extra headers for a response that was served after refusals: the trail, at most eight rows. */
function failureHeaders(failures: readonly UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extra: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extra);
  else sendJson(res, status, openAIError(message, type, null), extra);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
}

/** A refusal's code for the trail: the gateway's error code when it names one, else the HTTP status. */
function failureCode(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value)) {
      const err = isRecord(value.error) ? value.error : value;
      if (typeof err.code === 'string' && err.code !== '') return err.code;
      if (typeof err.type === 'string' && err.type !== '') return err.type;
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/** The answer that served a request: the model that really answered, its gateway response, and the refusals it cost. */
interface Served {
  model: ModelRow;
  upstream: Response;
  failures: UpstreamFailure[];
}

/**
 * Dial the ranked candidates in order until one answers. A refusal counts
 * against the model (three in a row spends it, and one sets it aside for the
 * next prompt's ranking) and goes on the trail; a fetch that never reached
 * the gateway counts the same way. Returns null when every candidate
 * refused: the prompt is lost.
 */
async function serve(makeBody: (model: string) => Json, requires: Requirement): Promise<Served | null> {
  const failures: UpstreamFailure[] = [];
  for (const model of candidatesFor(requires)) {
    const state = stateOf(model.id);
    state.calls += 1;
    let upstream: Response;
    try {
      upstream = await dial(model.id, makeBody(model.id));
    } catch {
      recordRefusal(model, failures, 0, 'fetch_error');
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      recordRefusal(model, failures, upstream.status, failureCode(upstream.status, text));
      continue;
    }
    state.consecutiveFailures = 0;
    return { model, upstream, failures };
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve(() => ({ ...body }), requirementsFor(body));
  if (result === null) return sendError(res, 'openai', 502, 'upstream_error', 'every free model that meets this request refused it');
  const { model, upstream, failures } = result;
  const extra = failureHeaders(failures);
  if (body.stream === true) {
    const stream = upstream.body;
    if (stream === null) return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with no body');
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...extra });
    try {
      for await (const frame of openAIStreamWithModel(stream, model.id)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model.id), extra);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve((id) => anthropicToOpenAI(body, id), requirementsFor(body));
  if (result === null) return sendError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused it');
  const { model, upstream, failures } = result;
  const extra = failureHeaders(failures);
  if (body.stream === true) {
    const stream = upstream.body;
    if (stream === null) return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with no body');
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...extra });
    try {
      for await (const event of anthropicEventsFromOpenAI(stream, model.id)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model.id), extra);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
