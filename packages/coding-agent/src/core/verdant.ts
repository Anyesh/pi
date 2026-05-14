import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, constants as fsConstants, statSync } from "node:fs";
import {
	access as fsAccess,
	mkdir as fsMkdir,
	readdir as fsReaddir,
	readFile as fsReadFile,
	stat as fsStat,
	writeFile as fsWriteFile,
} from "node:fs/promises";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import type {
	Api,
	AssistantMessage,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	Context,
	Model,
	SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import type { BashOperations } from "./tools/bash.js";
import type { EditOperations } from "./tools/edit.js";
import type { FindOperations } from "./tools/find.js";
import type { GrepOperations } from "./tools/grep.js";
import type { ToolsOptions } from "./tools/index.js";
import type { LsOperations } from "./tools/ls.js";
import type { ReadOperations } from "./tools/read.js";
import type { WriteOperations } from "./tools/write.js";

interface VerdantBindings {
	VerdantCache: new (storeDir: string) => NativeCache;
	deriveLlmKey: (chatRequestJson: string) => string;
	blake3File: (path: string) => string;
	matchBashRule: (rulesPath: string, command: string) => { matched: boolean; roots: string[]; env: string[] };
	compareVerify: (cached: Buffer, live: Buffer) => string | null;
}

interface NativeCache {
	lookup(key: string): Buffer | null;
	lookupRevalidate(key: string): Buffer | null;
	persist(key: string, bytes: Buffer, toolKind: string): void;
	persistWithFileRoots(
		key: string,
		bytes: Buffer,
		toolKind: string,
		fileRoots: { path: string; expectedHash: string }[],
	): void;
	invalidatePath(path: string): number;
	stats(): { entryCount: number; knownKinds: string[] };
	entryCount(): number;
}

export interface VerdantConfig {
	storeDir: string;
	bashRulesPath?: string;
	verify?: boolean;
	bindings: VerdantBindings;
	cache: NativeCache;
}

let _bindings: VerdantBindings | null | undefined;
let _config: VerdantConfig | null | undefined;

const esmRequire = createRequire(import.meta.url);

function loadVerdant(): VerdantBindings | null {
	if (_bindings !== undefined) return _bindings;
	try {
		_bindings = esmRequire("verdant-node") as VerdantBindings;
		return _bindings;
	} catch {
		_bindings = null;
		return null;
	}
}

export function getVerdantConfig(): VerdantConfig | null {
	if (_config !== undefined) return _config;
	const storeDir = process.env.VERDANT_STORE_DIR;
	if (!storeDir) {
		_config = null;
		return null;
	}
	const bindings = loadVerdant();
	if (!bindings) {
		console.warn("[verdant] VERDANT_STORE_DIR set but verdant-node not installed. Caching disabled.");
		_config = null;
		return null;
	}
	const cache = new bindings.VerdantCache(storeDir);
	_config = {
		storeDir,
		bashRulesPath: process.env.VERDANT_BASH_RULES,
		verify: process.env.VERDANT_VERIFY === "1",
		bindings,
		cache,
	};
	return _config;
}

function deriveToolKey(kind: string, ...parts: string[]): string {
	const h = createHash("sha256");
	h.update(kind);
	for (const p of parts) {
		h.update("\0");
		h.update(p);
	}
	return h.digest("hex");
}

function createCachedReadOperations(config: VerdantConfig): ReadOperations {
	const { cache, bindings } = config;

	return {
		async readFile(absolutePath) {
			const key = deriveToolKey("read", absolutePath);
			const cached = cache.lookupRevalidate(key);
			if (cached !== null) return cached;
			const real = await fsReadFile(absolutePath);
			const hash = bindings.blake3File(absolutePath);
			cache.persistWithFileRoots(key, real, "read", [{ path: absolutePath, expectedHash: hash }]);
			return real;
		},
		async access(absolutePath) {
			return fsAccess(absolutePath, fsConstants.R_OK);
		},
	};
}

function createCachedEditOperations(config: VerdantConfig): EditOperations {
	const { cache, bindings } = config;

	return {
		async readFile(absolutePath) {
			const key = deriveToolKey("read", absolutePath);
			const cached = cache.lookupRevalidate(key);
			if (cached !== null) return cached;
			const real = await fsReadFile(absolutePath);
			const hash = bindings.blake3File(absolutePath);
			cache.persistWithFileRoots(key, real, "read", [{ path: absolutePath, expectedHash: hash }]);
			return real;
		},
		async writeFile(absolutePath, content) {
			await fsWriteFile(absolutePath, content);
			cache.invalidatePath(absolutePath);
		},
		async access(absolutePath) {
			return fsAccess(absolutePath, fsConstants.R_OK);
		},
	};
}

function createCachedWriteOperations(config: VerdantConfig): WriteOperations {
	const { cache } = config;

	return {
		async writeFile(absolutePath, content) {
			await fsWriteFile(absolutePath, content);
			cache.invalidatePath(absolutePath);
		},
		async mkdir(dir) {
			await fsMkdir(dir, { recursive: true });
		},
	};
}

function createCachedGrepOperations(config: VerdantConfig): GrepOperations {
	const { cache, bindings } = config;

	return {
		async isDirectory(absolutePath) {
			try {
				const s = await fsStat(absolutePath);
				return s.isDirectory();
			} catch {
				return false;
			}
		},
		async readFile(absolutePath) {
			const key = deriveToolKey("read", absolutePath);
			const cached = cache.lookupRevalidate(key);
			if (cached !== null) return cached.toString("utf-8");
			const real = await fsReadFile(absolutePath, "utf-8");
			const hash = bindings.blake3File(absolutePath);
			cache.persistWithFileRoots(key, Buffer.from(real, "utf-8"), "read", [
				{ path: absolutePath, expectedHash: hash },
			]);
			return real;
		},
	};
}

function createCachedFindOperations(): FindOperations {
	return {
		async exists(absolutePath) {
			try {
				await fsAccess(absolutePath, fsConstants.F_OK);
				return true;
			} catch {
				return false;
			}
		},
		async glob(pattern, globCwd, globOpts) {
			const { glob: globFn } = await import("glob");
			return globFn(pattern, {
				cwd: globCwd,
				ignore: globOpts.ignore,
				...(globOpts.limit ? { maxResults: globOpts.limit } : {}),
			});
		},
	};
}

function createCachedLsOperations(): LsOperations {
	return {
		exists: (absolutePath) => existsSync(absolutePath),
		stat: (absolutePath) => statSync(absolutePath),
		readdir: (absolutePath) => fsReaddir(absolutePath),
	};
}

function createCachedBashOperations(config: VerdantConfig, origOps?: BashOperations): BashOperations {
	const { cache, bindings, bashRulesPath } = config;

	if (!bashRulesPath) {
		return origOps ?? defaultBashExec();
	}

	const rulesPath = bashRulesPath;

	return {
		async exec(command, execCwd, execOpts) {
			const ruleMatch = bindings.matchBashRule(rulesPath, command);

			if (!ruleMatch.matched) {
				if (origOps) return origOps.exec(command, execCwd, execOpts);
				return runShellCommand(command, execCwd, execOpts);
			}

			const envParts = ruleMatch.env.map((e: string) => `${e}=${process.env[e] ?? ""}`).sort();
			const key = deriveToolKey("bash", command, execCwd, ...envParts);
			const cached = cache.lookupRevalidate(key);
			if (cached !== null) {
				const payload = JSON.parse(cached.toString("utf-8"));
				if (execOpts.onData) execOpts.onData(Buffer.from(payload.stdout, "utf-8"));
				return { exitCode: payload.exitCode };
			}

			const result = await runShellCommand(command, execCwd, execOpts);
			const stdout = result._capturedStdout;

			const payloadJson = JSON.stringify({ stdout, exitCode: result.exitCode });
			const fileRoots: { path: string; expectedHash: string }[] = [];
			for (const root of ruleMatch.roots) {
				try {
					const rootPath = resolve(execCwd, root);
					const hash = bindings.blake3File(rootPath);
					fileRoots.push({ path: rootPath, expectedHash: hash });
				} catch {
					// skip roots that can't be hashed (globs expanding to multiple files)
				}
			}
			cache.persistWithFileRoots(key, Buffer.from(payloadJson, "utf-8"), "bash", fileRoots);
			return { exitCode: result.exitCode };
		},
	};
}

function runShellCommand(
	command: string,
	execCwd: string,
	execOpts: { onData?: (data: Buffer) => void; signal?: AbortSignal; timeout?: number; env?: NodeJS.ProcessEnv },
): Promise<{ exitCode: number | null; _capturedStdout: string }> {
	return new Promise((resolve, reject) => {
		const child = spawn(command, {
			cwd: execCwd,
			shell: true,
			env: execOpts.env ?? process.env,
			stdio: ["ignore", "pipe", "pipe"],
		});

		const chunks: Buffer[] = [];
		let timedOut = false;
		let timeoutHandle: NodeJS.Timeout | undefined;

		if (execOpts.timeout !== undefined && execOpts.timeout > 0) {
			timeoutHandle = setTimeout(() => {
				timedOut = true;
				child.kill("SIGTERM");
			}, execOpts.timeout * 1000);
		}

		child.stdout?.on("data", (data: Buffer) => {
			chunks.push(data);
			if (execOpts.onData) execOpts.onData(data);
		});
		child.stderr?.on("data", (data: Buffer) => {
			chunks.push(data);
			if (execOpts.onData) execOpts.onData(data);
		});

		const onAbort = () => child.kill("SIGTERM");
		if (execOpts.signal) {
			if (execOpts.signal.aborted) {
				onAbort();
			} else {
				execOpts.signal.addEventListener("abort", onAbort, { once: true });
			}
		}

		child.on("close", (code) => {
			if (timeoutHandle) clearTimeout(timeoutHandle);
			if (execOpts.signal) execOpts.signal.removeEventListener("abort", onAbort);
			if (execOpts.signal?.aborted) {
				reject(new Error("aborted"));
				return;
			}
			if (timedOut) {
				reject(new Error(`timeout:${execOpts.timeout}`));
				return;
			}
			const fullOutput = Buffer.concat(chunks).toString("utf-8");
			resolve({ exitCode: code, _capturedStdout: fullOutput });
		});

		child.on("error", (err) => {
			if (timeoutHandle) clearTimeout(timeoutHandle);
			reject(err);
		});
	});
}

function defaultBashExec(): BashOperations {
	return {
		exec: (command, cwd, opts) => runShellCommand(command, cwd, opts),
	};
}

export function mergeVerdantToolOptions(existing: ToolsOptions, config: VerdantConfig): ToolsOptions {
	return {
		read: {
			...existing.read,
			operations: createCachedReadOperations(config),
		},
		edit: {
			...existing.edit,
			operations: createCachedEditOperations(config),
		},
		write: {
			...existing.write,
			operations: createCachedWriteOperations(config),
		},
		grep: {
			...existing.grep,
			operations: createCachedGrepOperations(config),
		},
		find: {
			...existing.find,
			operations: createCachedFindOperations(),
		},
		ls: {
			...existing.ls,
			operations: createCachedLsOperations(),
		},
		bash: {
			...existing.bash,
			operations: createCachedBashOperations(config, existing.bash?.operations),
		},
	};
}

function synthesizeStream(msg: AssistantMessage): AssistantMessageEventStream {
	const events: AssistantMessageEvent[] = [];
	events.push({ type: "start", partial: msg });

	for (let i = 0; i < msg.content.length; i++) {
		const block = msg.content[i];
		if (block.type === "text") {
			events.push({ type: "text_start", contentIndex: i, partial: msg });
			events.push({ type: "text_delta", contentIndex: i, delta: block.text, partial: msg });
			events.push({ type: "text_end", contentIndex: i, content: block.text, partial: msg });
		} else if (block.type === "thinking") {
			events.push({ type: "thinking_start", contentIndex: i, partial: msg });
			events.push({ type: "thinking_delta", contentIndex: i, delta: block.thinking, partial: msg });
			events.push({ type: "thinking_end", contentIndex: i, content: block.thinking, partial: msg });
		} else if (block.type === "toolCall") {
			events.push({ type: "toolcall_start", contentIndex: i, partial: msg });
			events.push({ type: "toolcall_delta", contentIndex: i, delta: JSON.stringify(block.arguments), partial: msg });
			events.push({ type: "toolcall_end", contentIndex: i, toolCall: block, partial: msg });
		}
	}

	const stopReason = msg.stopReason;
	if (stopReason === "error" || stopReason === "aborted") {
		events.push({ type: "error", reason: stopReason, error: msg });
	} else {
		events.push({ type: "done", reason: stopReason, message: msg });
	}

	return {
		[Symbol.asyncIterator]() {
			let idx = 0;
			return {
				async next() {
					if (idx < events.length) {
						return { value: events[idx++], done: false };
					}
					return { value: undefined, done: true };
				},
			};
		},
		result() {
			return Promise.resolve(msg);
		},
		push() {},
		end() {},
	} as unknown as AssistantMessageEventStream;
}

interface ChatRequestMessage {
	role: string;
	content: unknown;
	tool_calls?: unknown[];
	tool_call_id?: string;
}

function buildChatRequestJson(model: Model<Api>, context: Context, options?: SimpleStreamOptions): string {
	const messages: ChatRequestMessage[] = [];

	if (context.systemPrompt) {
		messages.push({ role: "system", content: context.systemPrompt });
	}

	for (const msg of context.messages) {
		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				messages.push({ role: "user", content: msg.content });
			} else {
				const blocks = msg.content.map((b) => {
					if (b.type === "text") return { type: "text", text: b.text };
					return { type: "image_url", image_url: { url: (b as any).data } };
				});
				messages.push({ role: "user", content: blocks });
			}
		} else if (msg.role === "assistant") {
			const textBlocks: any[] = [];
			const toolCalls: any[] = [];
			for (const block of msg.content) {
				if (block.type === "text") {
					textBlocks.push({ type: "text", text: block.text });
				} else if (block.type === "toolCall") {
					toolCalls.push({
						id: block.id,
						type: "function",
						function: { name: block.name, arguments: JSON.stringify(block.arguments) },
					});
				}
			}
			const result: ChatRequestMessage = {
				role: "assistant",
				content: textBlocks.length > 0 ? textBlocks : null,
			};
			if (toolCalls.length > 0) result.tool_calls = toolCalls;
			messages.push(result);
		} else if (msg.role === "toolResult") {
			const content =
				msg.content.length === 1 && msg.content[0].type === "text"
					? msg.content[0].text
					: msg.content.map((b) => {
							if (b.type === "text") return { type: "text", text: b.text };
							return { type: "image_url", image_url: { url: (b as any).data } };
						});
			messages.push({ role: "tool", content, tool_call_id: msg.toolCallId });
		}
	}

	const req: Record<string, unknown> = {
		model: `${model.provider}/${model.id}`,
		messages,
	};

	if (context.tools && context.tools.length > 0) {
		req.tools = context.tools.map((t) => ({
			type: "function",
			function: { name: t.name, description: t.description, parameters: t.parameters },
		}));
	}

	if (options) {
		if ((options as any).temperature !== undefined) req.temperature = (options as any).temperature;
		if ((options as any).maxTokens !== undefined) req.max_tokens = (options as any).maxTokens;
	}

	return JSON.stringify(req);
}

export function wrapStreamFnWithVerdant(originalStreamFn: StreamFn, config: VerdantConfig): StreamFn {
	const { cache, bindings } = config;

	return function verdantCachedStream<TApi extends Api>(
		model: Model<TApi>,
		context: Context,
		opts?: SimpleStreamOptions,
	): AssistantMessageEventStream | Promise<AssistantMessageEventStream> {
		const reqJson = buildChatRequestJson(model, context, opts);
		let key: string;
		try {
			key = bindings.deriveLlmKey(reqJson);
		} catch {
			return originalStreamFn(model, context, opts);
		}

		const cached = cache.lookup(key);
		if (cached !== null) {
			const msg = JSON.parse(cached.toString("utf-8")) as AssistantMessage;
			return synthesizeStream(msg);
		}

		const rawResult = originalStreamFn(model, context, opts);

		function wrapStream(realStream: AssistantMessageEventStream): AssistantMessageEventStream {
			const wrappedIterator = {
				[Symbol.asyncIterator]() {
					const inner = realStream[Symbol.asyncIterator]();
					return {
						async next(): Promise<IteratorResult<AssistantMessageEvent>> {
							const result = await inner.next();
							if (!result.done) {
								const event = result.value;
								if (event.type === "done") {
									const bytes = Buffer.from(JSON.stringify(event.message), "utf-8");
									cache.persist(key, bytes, "llm_call");
								}
							}
							return result;
						},
					};
				},
				result() {
					return realStream.result();
				},
				push(event: AssistantMessageEvent) {
					realStream.push(event);
				},
				end(result?: AssistantMessage) {
					realStream.end(result);
				},
			};
			return wrappedIterator as unknown as AssistantMessageEventStream;
		}

		if (rawResult instanceof Promise) {
			return rawResult.then(wrapStream);
		}
		return wrapStream(rawResult);
	};
}

export function getVerdantStats(config: VerdantConfig): { entryCount: number; knownKinds: string[] } {
	return config.cache.stats();
}
