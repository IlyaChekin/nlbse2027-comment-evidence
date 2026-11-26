import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { createContext, Script, SourceTextModule, SyntheticModule } from "node:vm";

const here = fileURLToPath(new URL(".", import.meta.url));
const dir = resolve(here, "upstream/WeakSATD");
const manifest = JSON.parse(await readFile(resolve(here, "rules_manifest.json"), "utf8"));
const context = createContext(Object.create(null), {
  codeGeneration: { strings: false, wasm: false },
});
const cacheMitigations = !process.argv.includes("--uncached");

const published = ["478", "483", "242", "676", "415", "416", "783", "467", "401", "244",
                   "243", "135", "14", "690", "587", "781", "782", "560", "806"];
const only = process.argv.find(arg => arg.startsWith("--only="))?.slice(7).split(",");
if (only && only.some(rule => !published.includes(rule))) throw new Error("Unknown rule");
const active = only ? published.filter(rule => only.includes(rule)) : published;
const modules = new Map();

async function moduleFor(name) {
  if (name !== basename(name) || !Object.hasOwn(manifest.files, "WeakSATD/" + name)) {
    throw new Error("Unexpected rule dependency");
  }
  if (!modules.has(name)) {
    modules.set(name, (async () => {
      const path = resolve(dir, name);
      const data = await readFile(path);
      const digest = createHash("sha256").update(data).digest("hex");
      if (digest !== manifest.files["WeakSATD/" + name].sha256) {
        throw new Error("Rule dependency changed");
      }
      if (name.endsWith(".json")) {
        const parse = new Script("JSON.parse(" + JSON.stringify(data.toString("utf8")) + ")");
        const object = parse.runInContext(context);
        return new SyntheticModule(["default"], function () {
          this.setExport("default", object);
        }, { context, identifier: path });
      }
      // The original package uses the older JSON import-assertion syntax.
      let source = data.toString("utf8").replace(/ assert \{ type: 'json' \}/g, "");
      if (/\beval\(/.test(source)) {
        throw new Error("Unexpected dynamic evaluation in a scanner rule");
      }
      if (name === "findIssue.js" && cacheMitigations) {
        // Rules attach the lookup result for descriptions. They never mutate it or read it back.
        const entry = "export function getPotentialMitigations(issueNumber) {";
        if (!source.includes(entry)) throw new Error("Mitigation lookup signature changed");
        source = source.replace(entry, "function loadMitigations(issueNumber) {") + `
const mitigationCache = new Map();
export function getPotentialMitigations(issueNumber) {
  if (!mitigationCache.has(issueNumber)) {
    mitigationCache.set(issueNumber, loadMitigations(issueNumber));
  }
  return mitigationCache.get(issueNumber);
}
`;
      }
      return new SourceTextModule(source, { context, identifier: path });
    })());
  }
  return modules.get(name);
}

const functions = [];
for (const tag of active) {
  const rule = await moduleFor(`cwe_${tag}.js`);
  if (rule.status === "unlinked") {
    await rule.link(name => {
      if (!/^\.\/[A-Za-z_0-9.]+$/.test(name)) {
        throw new Error("Rule requested an external import");
      }
      return moduleFor(name.slice(2));
    });
  }
  await rule.evaluate({ timeout: 2000 });
  functions.push(rule.namespace.default);
}
context.rules = functions;

const utility = await readFile(resolve(here, "upstream/src/utility.js"), "utf8");
const utilityDigest = createHash("sha256").update(utility).digest("hex");
if (utilityDigest !== manifest.files["src/utility.js"].sha256) {
  throw new Error("Comment preprocessing source changed");
}
// Replay only the pure comment routines from the released annotation script.
const commentsRead = utility.slice(utility.indexOf("function extractComments"), utility.indexOf("// reads csv file"));
const commentsRemove = utility.slice(utility.indexOf("// extracts comments"), utility.indexOf("function checkIfFolderExists"));
new Script(commentsRead + "\n" + commentsRemove).runInContext(context);
const scan = new Script(`
JSON.stringify(JSON.parse(payload).map(([row, text]) => {
  const body = remove_comment_from_code(text, extract_comments(text)).split("\\n");
  const comments = extractComments(body);
  return [row, ...rules.map(check => {
    try {
      const lines = check(body.slice(), comments).lineNumbers;
      return Array.isArray(lines) ? Math.min(255,lines.length) : -1;
    } catch {
      return -1;
    }
  })]
}))
`);

function processRows(rows) {
  context.payload = JSON.stringify(rows);
  try {
    return JSON.parse(scan.runInContext(context, { timeout: 4000 }));
  } catch (error) {
    if (error.code !== "ERR_SCRIPT_EXECUTION_TIMEOUT") throw error;
    if (rows.length === 1) return [[rows[0][0], ...active.map(() => -1)]];
    const middle = Math.floor(rows.length / 2);
    return processRows(rows.slice(0, middle)).concat(processRows(rows.slice(middle)));
  }
}

console.log(["row", ...active.map(n => "rule_" + n)].join(","));
let pending = [];
function flush() {
  if (pending.length > 0) {
    const rows = processRows(pending);
    process.stdout.write(rows.map(row => row.join(",")).join("\n") + "\n");
    pending = [];
  }
}
const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of input) {
  pending.push(JSON.parse(line));
  if (pending.length === 128) flush();
}
flush();
