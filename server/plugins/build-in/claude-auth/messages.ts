import type { JsonValue, NetworkEventStream, PluginContext } from "cursor-byok:plugin";
import type { ModelSnapshot } from "cursor-byok:model";
import type {
  ForwardOutput,
  LlmContentPart,
  LlmRequest,
  ModelUsage,
  ProviderOutput,
} from "cursor-byok:provider";

const REPLAY_KIND = "claude_oauth";
const MESSAGES_URL = "https://api.anthropic.com/v1/messages";

/** Anthropic accepts Opus subscription calls only when this identity leads the system prompt. */
export const CLAUDE_CODE_SYSTEM_PROMPT =
  "You are Claude Code, Anthropic's official CLI for Claude.";

/** Anthropic's refusal for a prompt over the context window; the host compacts when it sees it. */
export const PROMPT_TOO_LONG = "prompt is too long";
/** Error bodies are a few hundred bytes; a larger body is not an Anthropic error object. */
const ERROR_BODY_LIMIT = 64 * 1024;

/** Safe classification only: upstream bodies are never retained, error text only when classified. */
export class HttpError extends Error {
  readonly headers: Record<string, string>;
  constructor(
    readonly status: number,
    headers: Record<string, string>,
    readonly errorType?: string,
    readonly detail?: string,
  ) {
    // `<label> <status>: <detail>` lets the host recognize rejected requests and overflow.
    super(`Anthropic Messages ${status}${detail ? `: ${detail}` : ""}`);
    this.name = "HttpError";
    // Retain retry timing, not arbitrary upstream headers that may contain credentials.
    this.headers = Object.fromEntries(
      Object.entries(headers)
        .filter(([name]) => name.toLowerCase() === "retry-after")
        .map(([name, value]) => [name.toLowerCase(), value]),
    );
  }
}

type ObjectValue = Record<string, JsonValue>;
function object(value: unknown): ObjectValue | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as ObjectValue
    : null;
}
function invalid(): never {
  throw new Error("Anthropic Messages returned an invalid or incomplete stream");
}
function string(value: unknown): string {
  if (typeof value !== "string") invalid();
  return value;
}
function count(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
}
function part(value: LlmContentPart): JsonValue {
  return value.type === "text" ? { type: "text", text: value.text } : {
    type: "image",
    source: { type: "base64", media_type: value.mediaType, data: value.dataBase64 },
  };
}
function signedThinking(value: JsonValue): boolean {
  const block = object(value);
  return !!block && ((block.type === "thinking" && typeof block.thinking === "string" &&
    typeof block.signature === "string" && block.signature.length > 0) ||
    (block.type === "redacted_thinking" && typeof block.data === "string"));
}
/** Anthropic accepts only `[a-zA-Z0-9_-]+` ids; calls made by other models may contain more. */
function toolUseId(callId: string): string {
  return callId.replace(/[^a-zA-Z0-9_-]/g, "_");
}

export function buildMessagesBody(model: ModelSnapshot, request: LlmRequest): ObjectValue {
  const messages: JsonValue[] = request.messages.map((message): JsonValue => {
    if (message.role === "assistant") {
      const replay = message.replayState;
      const blocks = object(replay?.value)?.blocks;
      let content: JsonValue[];
      if (replay?.providerKind === REPLAY_KIND) {
        if (!Array.isArray(blocks)) throw new Error("Invalid Claude OAuth replay state");
        // Full ordered blocks are authoritative, including text between thinking and tools.
        content = blocks;
      } else {
        content = replay?.providerKind === "anthropic" && Array.isArray(blocks)
          ? blocks.filter(signedThinking)
          : [];
        if (message.text) content.push({ type: "text", text: message.text });
        content.push(...message.toolCalls.map((tool): JsonValue => ({
          type: "tool_use",
          id: toolUseId(tool.callId),
          name: tool.name,
          input: tool.arguments,
        })));
      }
      return { role: "assistant", content };
    }
    if (message.role === "tool") {
      return {
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: toolUseId(message.callId),
          is_error: message.isError,
          content: message.parts.length ? message.parts.map(part) : message.content,
        }],
      };
    }
    // Historical system messages are conversation context, not stable instructions.
    return { role: "user", content: message.content.map(part) };
  });
  const maximum = count(model.maxOutputTokens);
  const requested = count(request.maxOutputTokens);
  const maxTokens = Math.min(
    requested && requested > 0 ? requested : 8192,
    maximum && maximum > 0 ? maximum : Number.MAX_SAFE_INTEGER,
  );
  const system: JsonValue[] = [{ type: "text", text: CLAUDE_CODE_SYSTEM_PROMPT }];
  if (request.instructions.length > 0) {
    system.push({ type: "text", text: request.instructions });
  }
  const body: ObjectValue = {
    model: model.id,
    system,
    messages,
    max_tokens: maxTokens,
    stream: true,
    cache_control: { type: "ephemeral" },
  };
  if (request.tools.length) {
    body.tools = request.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters,
    }));
  }
  const metadata = object(model.privateData);
  if (request.reasoning.enabled && metadata?.thinking === "adaptive") {
    body.thinking = { type: "adaptive" };
    if (
      request.reasoning.effort && Array.isArray(metadata.efforts) &&
      metadata.efforts.includes(request.reasoning.effort)
    ) {
      body.output_config = { effort: request.reasoning.effort };
    }
  } else if (request.reasoning.enabled && metadata?.thinking === "enabled" && maxTokens > 1024) {
    body.thinking = {
      type: "enabled",
      budget_tokens: Math.min(8192, Math.max(1024, Math.floor(maxTokens / 2)), maxTokens - 1),
    };
  }
  return body;
}

/** Race host operations too: cancellation must not wait for another upstream line. */
function abortable<T>(operation: Promise<T>, signal: AbortSignal): Promise<T> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => {
      cleanup();
      reject(signal.reason);
    };
    const cleanup = () => signal.removeEventListener("abort", abort);
    signal.addEventListener("abort", abort, { once: true });
    operation.then((result) => {
      cleanup();
      resolve(result);
    }, (error) => {
      cleanup();
      reject(error);
    });
  });
}

async function* events(
  lines: AsyncIterable<string>,
  signal: AbortSignal,
): AsyncGenerator<ObjectValue> {
  const iterator = lines[Symbol.asyncIterator]();
  let data: string[] = [];
  let eventName = "";
  try {
    while (true) {
      signal.throwIfAborted();
      const item = await abortable(iterator.next(), signal);
      signal.throwIfAborted();
      if (item.done) break;
      const line = item.value.replace(/\r$/, "");
      if (line === "") {
        if (data.length) {
          let value: ObjectValue | null;
          try {
            value = object(JSON.parse(data.join("\n")));
          } catch {
            invalid();
          }
          if (
            !value || typeof value.type !== "string" ||
            (eventName && eventName !== "message" && eventName !== value.type)
          ) invalid();
          yield value;
        }
        data = [];
        eventName = "";
      } else if (!line.startsWith(":")) {
        const colon = line.indexOf(":");
        const field = colon < 0 ? line : line.slice(0, colon);
        const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
        if (field === "data") data.push(value);
        else if (field === "event") eventName = value;
      }
    }
    // An unterminated SSE event is not a terminal message, even if its JSON looks complete.
    if (data.length) invalid();
  } finally {
    // Returning a pending async generator can itself hang after cancellation.
    try {
      void iterator.return?.().catch(() => {});
    } catch { /* Preserve original failure. */ }
  }
}

type BlockState = { value: ObjectValue; arguments: string; hasArguments: boolean };
function mergeUsage(usage: ModelUsage, value: JsonValue | undefined): void {
  const update = object(value);
  if (!update) invalid();
  const fields = {
    input_tokens: "inputTokens",
    output_tokens: "outputTokens",
    cache_read_input_tokens: "cacheReadTokens",
    cache_creation_input_tokens: "cacheWriteTokens",
  } as const;
  for (const [upstream, local] of Object.entries(fields)) {
    if (update[upstream] !== undefined) {
      const next = count(update[upstream]);
      if (next === null) invalid();
      usage[local] = next;
    }
  }
  if (usage.inputTokens !== null) {
    // Anthropic's input_tokens excludes the cache; the whole prompt drives host compaction.
    usage.contextInputTokens = usage.inputTokens + (usage.cacheReadTokens ?? 0) +
      (usage.cacheWriteTokens ?? 0);
    if (usage.outputTokens !== null) {
      usage.totalTokens = usage.contextInputTokens + usage.outputTokens;
    }
  }
}
function streamError(value: ObjectValue, headers: Record<string, string>): HttpError {
  const errorType = object(value.error)?.type;
  const statuses: Record<string, number> = {
    invalid_request_error: 400,
    authentication_error: 401,
    permission_error: 403,
    not_found_error: 404,
    request_too_large: 413,
    rate_limit_error: 429,
    api_error: 500,
    overloaded_error: 529,
  };
  const known = typeof errorType === "string" && Object.hasOwn(statuses, errorType)
    ? errorType
    : "api_error";
  return new HttpError(statuses[known], headers, known);
}

/**
 * Keeps the over-limit classification, or a request-validation message that names the rejected
 * request field (`tools.35.custom.input_schema: ...`). Any other upstream text is dropped.
 */
function rejectionDetail(body: string): string | undefined {
  let message: unknown;
  try {
    message = object(object(JSON.parse(body))?.error)?.message;
  } catch {
    return undefined;
  }
  if (typeof message !== "string") return undefined;
  if (message.toLowerCase().includes(PROMPT_TOO_LONG)) return PROMPT_TOO_LONG;
  return /^[a-z_]+(\.[A-Za-z0-9_-]+)*: [^\n]{1,300}$/.test(message) ? message : undefined;
}

/** Reads a bounded error body and drains the host's lazy iterator, which would stay open. */
async function errorBody(response: NetworkEventStream, signal: AbortSignal): Promise<string> {
  const iterator = response.lines[Symbol.asyncIterator]();
  let body = "";
  try {
    while (body.length <= ERROR_BODY_LIMIT) {
      const item = await abortable(iterator.next(), signal);
      if (item.done) break;
      body += item.value;
    }
  } catch {
    signal.throwIfAborted();
  } finally {
    try {
      void iterator.return?.().catch(() => {});
    } catch { /* Preserve HTTP status. */ }
  }
  return body;
}

function betas(value: string | undefined): string[] {
  return (value ?? "").split(",").map((beta) => beta.trim()).filter(Boolean);
}

/**
 * The client's request with the upstream model id and the identity block the subscription
 * requires in front of the client's own system prompt. Everything else is the client's.
 */
export function forwardBody(model: ModelSnapshot, body: ObjectValue): ObjectValue {
  const client = body.system;
  const blocks: JsonValue[] = Array.isArray(client)
    ? client
    : typeof client === "string"
    ? client ? [{ type: "text", text: client }] : []
    : client === undefined || client === null
    ? []
    : [client];
  const first = object(blocks[0]);
  const system = first?.type === "text" && first.text === CLAUDE_CODE_SYSTEM_PROMPT
    ? blocks
    : [{ type: "text", text: CLAUDE_CODE_SYSTEM_PROMPT }, ...blocks];
  return { ...body, model: model.id, system };
}

/** Account headers, with the client's API version and its betas added to the account's. */
export function forwardHeaders(
  account: Record<string, string>,
  client: Record<string, string>,
  stream: boolean,
): Record<string, string> {
  return {
    ...account,
    accept: stream ? "text/event-stream" : "application/json",
    "content-type": "application/json",
    "anthropic-version": client["anthropic-version"] || account["anthropic-version"],
    "anthropic-beta": [
      ...new Set([...betas(account["anthropic-beta"]), ...betas(client["anthropic-beta"])]),
    ].join(","),
  };
}

/**
 * Sends a client's Messages request and relays the upstream response unchanged, errors
 * included. A 401 is withheld so the caller can refresh the token and retry once.
 */
export async function forwardMessages(
  model: ModelSnapshot,
  body: ObjectValue,
  clientHeaders: Record<string, string>,
  accountHeaders: Record<string, string>,
  output: ForwardOutput,
  context: PluginContext,
): Promise<void> {
  context.signal.throwIfAborted();
  const response = await abortable(
    context.network.stream(MESSAGES_URL, {
      method: "POST",
      headers: forwardHeaders(accountHeaders, clientHeaders, body.stream === true),
      body: JSON.stringify(forwardBody(model, body)),
    }),
    context.signal,
  );
  context.signal.throwIfAborted();
  if (response.status === 401) {
    const text = await errorBody(response, context.signal);
    throw new HttpError(401, response.headers, undefined, rejectionDetail(text));
  }
  output.emit({ type: "response-start", status: response.status, headers: response.headers });
  const iterator = response.lines[Symbol.asyncIterator]();
  try {
    while (true) {
      const item = await abortable(iterator.next(), context.signal);
      if (item.done) break;
      output.emit({ type: "response-line", line: item.value });
    }
  } finally {
    try {
      void iterator.return?.().catch(() => {});
    } catch { /* Preserve the relay outcome. */ }
  }
  // The client already has the upstream error. Only a rate limit describes the shared account;
  // any other rejection (400, 403, 404, ...) can come from this client's own request content.
  if (response.status === 429) throw new HttpError(429, response.headers);
  if (response.status < 200 || response.status >= 300) {
    throw new RelayedRejection(response.status);
  }
}

/** An upstream rejection already relayed to the external client; the account stays as it is. */
export class RelayedRejection extends Error {
  constructor(readonly status: number) {
    super(`Anthropic Messages ${status}`);
    this.name = "RelayedRejection";
  }
}

export async function streamMessages(
  model: ModelSnapshot,
  request: LlmRequest,
  headers: Record<string, string>,
  output: ProviderOutput,
  context: PluginContext,
): Promise<void> {
  context.signal.throwIfAborted();
  const response = await abortable(
    context.network.stream(MESSAGES_URL, {
      method: "POST",
      headers: {
        ...headers,
        accept: "text/event-stream",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(buildMessagesBody(model, request)),
    }),
    context.signal,
  );
  context.signal.throwIfAborted();
  if (response.status < 200 || response.status >= 300) {
    const body = await errorBody(response, context.signal);
    throw new HttpError(response.status, response.headers, undefined, rejectionDetail(body));
  }
  const blocks: ObjectValue[] = [];
  let active: BlockState | null = null;
  let activeIndex = -1;
  let started = false;
  let sawTool = false;
  let finishing = false;
  let finish: "stop" | "length" | "tool-use" | null = null;
  let hasUsage = false;
  const usage: ModelUsage = {
    inputTokens: null,
    contextInputTokens: null,
    outputTokens: null,
    totalTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    reasoningTokens: null,
  };
  for await (const event of events(response.lines, context.signal)) {
    context.signal.throwIfAborted();
    if (event.type === "ping") continue;
    if (event.type === "error") throw streamError(event, response.headers);
    if (event.type === "message_start") {
      if (started || !object(event.message)) invalid();
      started = true;
      const initial = object(event.message)!;
      if (
        initial.content !== undefined &&
        (!Array.isArray(initial.content) || initial.content.length !== 0)
      ) invalid();
      if (initial.usage !== undefined) {
        mergeUsage(usage, initial.usage);
        hasUsage = true;
      }
      continue;
    }
    if (!started) invalid();
    switch (event.type) {
      case "content_block_start": {
        const index = count(event.index);
        const value = object(event.content_block);
        if (active || finishing || index === null || index !== blocks.length || !value) invalid();
        activeIndex = index;
        active = { value: { ...value }, arguments: "", hasArguments: false };
        switch (value.type) {
          case "text":
            string(value.text);
            output.emit({ type: "text-start" });
            if (value.text) output.emit({ type: "text-delta", text: string(value.text) });
            break;
          case "thinking":
            string(value.thinking);
            if (value.signature !== undefined) string(value.signature);
            active.value.signature ??= "";
            output.emit({ type: "thinking-start" });
            if (value.thinking) {
              output.emit({ type: "thinking-delta", text: string(value.thinking) });
            }
            break;
          case "redacted_thinking":
            string(value.data);
            break;
          case "tool_use":
            if (!string(value.id) || !string(value.name) || !object(value.input)) invalid();
            sawTool = true;
            output.emit({
              type: "tool-call-start",
              index,
              callId: string(value.id),
              name: string(value.name),
            });
            break;
          default:
            invalid();
        }
        break;
      }
      case "content_block_delta": {
        if (!active || event.index !== activeIndex) invalid();
        const delta = object(event.delta);
        if (!delta) invalid();
        const block = active.value;
        if (block.type === "text" && delta.type === "text_delta") {
          const text = string(delta.text);
          block.text = string(block.text) + text;
          output.emit({ type: "text-delta", text });
        } else if (block.type === "thinking" && delta.type === "thinking_delta") {
          const text = string(delta.thinking);
          block.thinking = string(block.thinking) + text;
          output.emit({ type: "thinking-delta", text });
        } else if (block.type === "thinking" && delta.type === "signature_delta") {
          block.signature = string(block.signature) + string(delta.signature);
        } else if (block.type === "tool_use" && delta.type === "input_json_delta") {
          const text = string(delta.partial_json);
          active.arguments += text;
          if (text) active.hasArguments = true;
          output.emit({ type: "tool-call-arguments-delta", index: activeIndex, delta: text });
        } else invalid();
        break;
      }
      case "content_block_stop": {
        if (!active || event.index !== activeIndex) invalid();
        const block = active.value;
        if (block.type === "text") output.emit({ type: "text-end" });
        else if (block.type === "thinking") {
          if (!signedThinking(block)) invalid();
          output.emit({ type: "thinking-end" });
        } else if (block.type === "tool_use") {
          if (active.hasArguments) {
            try {
              block.input = JSON.parse(active.arguments);
            } catch {
              invalid();
            }
            if (!object(block.input)) invalid();
          } else {
            output.emit({
              type: "tool-call-arguments-delta",
              index: activeIndex,
              delta: JSON.stringify(block.input),
            });
          }
          output.emit({ type: "tool-call-end", index: activeIndex });
        }
        blocks.push(block);
        active = null;
        break;
      }
      case "message_delta": {
        if (active) invalid();
        finishing = true;
        const delta = object(event.delta);
        if (!delta) invalid();
        if (event.usage !== undefined) {
          mergeUsage(usage, event.usage);
          hasUsage = true;
        }
        if (delta.stop_reason !== undefined && delta.stop_reason !== null) {
          switch (delta.stop_reason) {
            case "max_tokens":
            case "model_context_window_exceeded":
              finish = "length";
              break;
            case "tool_use":
              finish = "tool-use";
              break;
            case "end_turn":
            case "stop_sequence":
            case "pause_turn":
            case "refusal":
              finish = "stop";
              break;
            default:
              invalid();
          }
        }
        break;
      }
      case "message_stop":
        if (active || !finish) invalid();
        output.emit({ type: "replay-state", providerKind: REPLAY_KIND, value: { blocks } });
        if (hasUsage) output.emit({ type: "usage", usage });
        context.signal.throwIfAborted();
        output.emit({
          type: "done",
          reason: finish === "length" ? finish : sawTool ? "tool-use" : finish,
        });
        return;
      default:
        invalid();
    }
  }
  context.signal.throwIfAborted();
  invalid();
}
