import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(app, 'node_modules/.tmp/spaceflow-tests');
mkdirSync(output, { recursive: true });
writeFileSync(path.join(output, 'package.json'), '{"type":"commonjs"}\n');

// Use the existing TypeScript compiler and Node test runner; no test dependency.
const compile = spawnSync(process.execPath, [
  'node_modules/typescript/bin/tsc', '--module', 'commonjs',
  '--moduleResolution', 'node', '--target', 'ES2023', '--esModuleInterop',
  '--skipLibCheck', '--strict', '--outDir', output,
  'tests/npz_metadata.test.ts',
], { cwd: app, stdio: 'inherit' });
if (compile.error) throw compile.error;
if (compile.status !== 0) process.exit(compile.status ?? 1);
const result = spawnSync(process.execPath, [
  '--test', path.join(output, 'tests/npz_metadata.test.js'),
], { cwd: app, stdio: 'inherit' });
if (result.error) throw result.error;
process.exit(result.status ?? 1);
