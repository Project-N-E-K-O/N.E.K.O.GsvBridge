# Voice editor source and build

The repository originally shipped only a prebuilt Nuxt application. The voice editor
now has editable Vue source in `src/VoiceEditor.vue`, with model/default handling in
`src/voice-editor.js`. Other pages continue to use the existing distribution.

```sh
npm ci
npm test
npm run build
```

The SFC compiler and esbuild are development dependencies only. The build emits a
content-hashed editor chunk and installs an import map before Nuxt's module scripts
in every generated HTML entry. This replaces the original editor route without
editing the compressed framework or other page bundles.

`src/nuxt-runtime.js` maps the Vue and Nuxt exports of this repository's distributed
`CyWKjLMZ.js` runtime. Sharing that instance is required for reactivity, routing and
i18n. When replacing the entire Nuxt distribution, update this adapter and the legacy
editor route in `scripts/build.mjs`, then run the browser regression suite. Do not
patch the compressed route directly.

V5 Dev/Turbo selection rescans and selects compatible weights. Automatic sampling
steps and CFG stay `null` in API requests, while the page displays the selected
model's effective defaults. Explicit values, including old profiles, remain explicit.
Saving and reopening a profile preserves its model, reference audio and parameter
choices. The examples and tests use generic/temporary data, not machine voice files.

The optional browser regression test is in `tests/test_voice_editor_browser.py` at
the repository root. Enable it with `GSV_BROWSER_TEST=1`; Playwright is required in
the development test environment. `GSV_BROWSER_EXECUTABLE` can point to an installed
Chrome/Chromium executable. The test starts a temporary local server and uses a
temporary voice directory. It does not write to the user's running Bridge service.

This browser acceptance layer checks the generated page, V5 options, weight
selection, automatic/custom parameters, and persistence through the configuration
API. Model inference is a separate opt-in host test; it does not silently count a
CPU fallback as a successful GPU regression.
