import { readFile, writeFile, readdir, unlink } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';
import { build } from 'esbuild';

const root = fileURLToPath(new URL('../', import.meta.url));
const inputs = ['src/VoiceEditor.vue', 'src/voice-editor.js', 'src/nuxt-runtime.js', 'scripts/build.mjs', 'package-lock.json'];
const sourceHash = createHash('sha256');
for (const input of inputs) sourceHash.update((await readFile(path.join(root, input), 'utf8')).replaceAll('\r\n', '\n'));
const sourcePath = path.join(root, 'src/VoiceEditor.vue');
const source = await readFile(sourcePath, 'utf8');
const { descriptor, errors } = parse(source, { filename: sourcePath });
if (errors.length) throw errors[0];
const script = compileScript(descriptor, { id: 'gsv-voice-editor', genDefaultAs: '__page' });
const template = compileTemplate({
  source: descriptor.template.content, filename: sourcePath, id: 'gsv-voice-editor',
  compilerOptions: { bindingMetadata: script.bindings, hoistStatic: false },
});
if (template.errors.length) throw template.errors[0];
const result = await build({
  stdin: { contents: `${script.content}\n${template.code}\n__page.render = render; export default __page;`,
    resolveDir: path.join(root, 'src'), sourcefile: 'VoiceEditor.vue.js' },
  bundle: true, format: 'esm', minify: true, target: 'es2022', write: false,
  alias: { vue: path.join(root, 'src/nuxt-runtime.js') },
  external: ['./CyWKjLMZ.js', './CTDpj6JD.js', './C-oSG0qo.js'],
});
const output = result.outputFiles[0].text;
const hash = createHash('sha256').update(output).digest('hex').slice(0, 12);
const name = `voice-editor.${hash}.js`;
const publicRoot = path.join(root, '.output/public');
const manifestPath = path.join(publicRoot, '_nuxt/voice-editor-manifest.json');
let previous;
try { previous = JSON.parse(await readFile(manifestPath, 'utf8')); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
await writeFile(path.join(publicRoot, '_nuxt', name), output);
const map = { imports: { '/_nuxt/C-I8AjUq.js': `/_nuxt/${name}` } };
const tag = `<script type="importmap" data-gsv-voice-editor>${JSON.stringify(map)}</script>`;
async function updateHtml(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.isDirectory()) await updateHtml(path.join(directory, entry.name));
    else if (entry.name.endsWith('.html')) {
      const file = path.join(directory, entry.name);
      const html = (await readFile(file, 'utf8')).replace(/<script type="importmap" data-gsv-voice-editor>.*?<\/script>/g, '');
      await writeFile(file, html.replace('<head>', `<head>${tag}`));
    }
  }
}
await updateHtml(publicRoot);
await writeFile(manifestPath, JSON.stringify({ entry: name, legacyRoute: 'C-I8AjUq.js', inputs, sourceSHA256: sourceHash.digest('hex') }, null, 2) + '\n');
if (previous?.entry !== name && /^voice-editor\.[a-f0-9]{12}\.js$/.test(previous?.entry || '')) {
  await unlink(path.join(publicRoot, '_nuxt', previous.entry));
}
console.log(`Built ${name}`);
