// Protocol 1: one structured turn; OpenCode owns configuration, policy and auth.
// The pinned SDK supplies the assembled host. No CLI credential files are read here.
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const VERSION = '2.0.32';
const root = process.env.ONE_OPENCODE_SDK_DIR;
const load = (pkg, file) => import(pathToFileURL(path.join(root, 'node_modules', pkg, 'dist', file)).href);
let failure = 'sdk_unavailable';
try {
  if (!root) throw new Error();
  for (const pkg of ['sdk', 'core', 'ai']) {
    const manifest = JSON.parse(await readFile(path.join(root, 'node_modules', '@opencode', pkg, 'package.json'), 'utf8'));
    if (manifest.version !== VERSION) throw new Error();
  }
  const { OpenCode } = await load('@opencode/sdk', 'effect/index.js');
  const { makeLocationNode } = await load('@opencode/util', 'effect/app-node.js');
  const { llmClient } = await load('@opencode/core', 'effect/app-node-platform.js');
  const { Generate } = await load('@opencode/core', 'generate.js');
  const { ModelResolver } = await load('@opencode/core', 'model-resolver.js');
  const { Config } = await load('@opencode/core', 'config.js');
  const { ManagedPolicy } = await load('@opencode/core', 'managed-policy.js');
  const { Model } = await load('@opencode/core', 'model.js');
  const { LLM, LLMClient } = await load('@opencode/ai', 'index.js');
  const { Effect, Layer } = await load('effect', 'index.js');
  failure = 'invalid_request';
  const input = JSON.parse(await new Response(Bun.stdin.stream()).text());
  if (input.version !== 1 || !Array.isArray(input.messages) || !input.model?.providerID || !input.model?.id) throw new Error();
  const calls = new Map();
  const messages = input.messages.map((message) => {
    if (message.role === 'assistant' && message.opencode_message) {
      for (const part of message.opencode_message.content) if (part.type === 'tool-call') calls.set(part.id, part.name);
      return message.opencode_message;
    }
    if (message.role === 'tool') {
      const name = calls.get(message.tool_call_id);
      if (!name) throw new Error();
      return {role:'tool', content:[{type:'tool-result', id:message.tool_call_id, name,
        result:{type:'text', value:message.content}}]};
    }
    const content = typeof message.content === 'string' ? [{type:'text',text:message.content}] : message.content ?? [];
    // Fail closed on media or opaque history this protocol cannot represent.
    if (content.some(part => part.type !== 'text')) throw new Error();
    for (const call of message.tool_calls ?? []) {
      calls.set(call.id, call.function.name);
      content.push({type:'tool-call',id:call.id,name:call.function.name,input:JSON.parse(call.function.arguments)});
    }
    return {role:message.role,content};
  });
  if (input.max_tokens !== null && (!Number.isInteger(input.max_tokens) || input.max_tokens <= 0)) throw new Error();
  const layer = Layer.effect(Generate.Service, Effect.gen(function* () {
    const resolver = yield* ModelResolver.Service;
    const models = yield* Model.Service;
    const config = yield* Config.Service;
    const managed = yield* ManagedPolicy.Service;
    const llm = yield* LLMClient.Service;
    return {text: () => Effect.gen(function* () {
      failure = 'policy_denied';
      const policies = ManagedPolicy.statements(yield* config.entries(), managed.current());
      // one-cli owns tool execution, so OpenCode permission hooks cannot inspect
      // its resources. Refuse restrictive policies rather than bypassing them.
      if ((input.tools?.length || messages.some(message => message.role === 'tool')) &&
          policies.some(policy => policy.effect === 'deny' && policy.action !== 'provider.use')) throw new Error();
      failure = 'model_unavailable';
      const available = yield* models.available();
      if (!available.some(model => model.providerID === input.model.providerID && model.id === input.model.id)) throw new Error();
      const selected = yield* resolver.resolve(input.model);
      if (!selected) throw new Error();
      failure = 'invalid_bounds';
      if (input.max_tokens && selected.limit?.output && input.max_tokens > selected.limit.output) throw new Error();
      failure = 'generation_failed';
      const response = yield* llm.generate(LLM.request({
        model:selected.model, messages,
        tools:(input.tools ?? []).map(tool => ({name:tool.function.name,
          description:tool.function.description ?? '',inputSchema:tool.function.parameters})),
        toolChoice: input.tools?.length ? 'auto' : 'none',
        generation:{maxTokens:input.max_tokens ?? undefined,temperature:input.temperature ?? undefined},
        http:{headers:{'x-opencode-session':input.session_id}},
      }));
      return JSON.stringify({version:1,text:response.text,thinking:response.reasoning,
        message:response.message,tool_calls:response.toolCalls,usage:response.usage ?? {},
        finish_reason:response.finishReason.normalized === 'tool-calls' ? 'tool_calls' : response.finishReason.normalized});
    })};
  }));
  const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const host = yield* OpenCode.create({
      config:{directory:process.cwd(),content:process.env.OPENCODE_CONFIG_CONTENT,
        project:process.env.OPENCODE_CONFIG_PROJECT_DISABLE !== '1',
        claude:process.env.OPENCODE_DISABLE_CLAUDE_CODE !== '1'},
      models:{fetch:process.env.OPENCODE_DISABLE_MODELS_FETCH !== '1'},fs:{filewatcher:false},
    }, {overrides:[Generate.node.replace(makeLocationNode({service:Generate.Service,layer,
      deps:[ModelResolver.node, Model.node, Config.node, ManagedPolicy.node, llmClient]}))]});
    return yield* host.generate.text({prompt:'one-cli structured turn',model:input.model});
  })));
  process.stdout.write(result.text + '\n');
} catch {
  // Never expose upstream error bodies, headers, config values or SDK stacks.
  process.stdout.write(JSON.stringify({version:1,error:failure}) + '\n');
  process.exitCode = 1;
}
