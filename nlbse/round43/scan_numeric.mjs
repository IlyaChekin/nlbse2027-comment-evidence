import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { createContext, Script, SourceTextModule, SyntheticModule } from "node:vm";

const here = fileURLToPath(new URL(".", import.meta.url));
const assets = resolve(here, process.env.NLBSE_WEAK_SOURCE || "../round19/weak_satd");
const dir = resolve(assets, "upstream/WeakSATD");
const manifest = JSON.parse(await readFile(resolve(assets, "rules_manifest.json"), "utf8"));
const context = createContext(Object.create(null), {
  codeGeneration: { strings: false, wasm: false },
});
const cacheMitigations = !process.argv.includes("--uncached");
const numeric = true;

const active = ["676", "783", "467", "478", "483", "587", "195", "14", "244", "782", "690", "401", "242", "415", "416", "243", "135", "781", "560", "806"];
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
      if (name === "cwe_195.js" && numeric) {
        const expression = "let result = eval(right)";
        if (source.split(expression).length !== 2) throw new Error("Numeric rule signature changed");
        source = source.replace(expression, "let result = boundedNumber(right)") + `
function boundedNumber(text) {
  if (text.length > 64) throw new SyntaxError("Expected a numeric literal");
  const match = /^([+-]?)(0[xX][a-fA-F0-9]+|0[bB][01]+|0[oO][0-7]+|(?:0|[1-9][0-9]*)(?:\\.[0-9]*)?(?:[eE][+-]?[0-9]+)?|\\.[0-9]+(?:[eE][+-]?[0-9]+)?)$/.exec(text);
  if (!match) throw new SyntaxError("Expected a numeric literal");
  const number = Number(match[2]) * (match[1] === "-" ? -1 : 1);
  if (!Number.isFinite(number)) throw new SyntaxError("Expected a finite number");
  return number;
}
`;
      }
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
      if (name === "cwe_676.js" && !process.argv.includes("--original-functions")) {
        const legacy = "const findFunctions = (data, functionList) => {";
        if (!source.includes(legacy)) throw new Error("Function matcher signature changed");
        source = source.replace(legacy, "const findRegExpFunctions = (data, functionList) => {") + `
const literalIndexes = new WeakMap();
const findFunctions = (data, names) => {
  if (!literalIndexes.has(names)) {
    const literal = /^[A-Za-z_][A-Za-z0-9_]*$/;
    let index = null;
    if (names.every(name => typeof name === "string" && literal.test(name))) {
      index = new Map();
      for (let j = 0; j < names.length; j++) {
        const positions = index.get(names[j]) || [];
        positions.push(j);
        index.set(names[j], positions);
      }
    }
    literalIndexes.set(names, index);
  }
  const index = literalIndexes.get(names);
  if (!index) return findRegExpFunctions(data, names);
  const result = [];
  const call = /(?:^|[\\s;])([A-Za-z_][A-Za-z0-9_]*)\\s*\\(/g;
  for (let i = 0; i < data.length; i++) {
    const positions = new Set();
    for (const match of data[i].matchAll(call)) {
      for (const j of index.get(match[1]) || []) positions.add(j);
    }
    for (const j of [...positions].sort((a,b) => a-b)) {
      result.push({ lineNumber: i+1, functionName: names[j] });
    }
  }
  return result;
};
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

const utility = await readFile(resolve(assets, "upstream/src/utility.js"), "utf8");
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
  let available = true;
  for (const [j, check] of rules.entries()) {
    try {
      const lines = check(body.slice(), comments).lineNumbers;
      if (!Array.isArray(lines)) available = false;
      else if (lines.length > 0) {
        if (j !== 1) return [row, 1, j + 1, lines.length, 0, 0, 0, 0, 0, body.length];
        if (lines.length === 1) {
          const line = lines[0];
          const good = Number.isSafeInteger(line) && line >= 1 && line <= body.length;
          const k = good ? line : 0;
          const seen = Number(good);
          return [row, 1, j + 1, 1, k, k, seen, 0, seen, body.length];
        }
        const valid = lines.filter(line => Number.isSafeInteger(line) && line >= 1 && line <= body.length);
        const unique = [...new Set(valid)].sort((a, b) => a - b);
        let gap = 0;
        for (let k = 1; k < unique.length; k++) {
          const d = unique[k] - unique[k - 1];
          gap = gap ? Math.min(gap, d) : d;
        }
        return [row, 1, j + 1, lines.length, unique[0] || 0, unique[unique.length - 1] || 0,
                unique.length, gap, valid.length, body.length];
      }
    } catch {
      available = false;
    }
  }
  return [row, available ? 0 : -1, 0, 0, 0, 0, 0, 0, 0, body.length];
}))
`);

function processRows(rows) {
  context.payload = JSON.stringify(rows);
  try {
    return JSON.parse(scan.runInContext(context, { timeout: 4000 }));
  } catch (error) {
    if (error.code !== "ERR_SCRIPT_EXECUTION_TIMEOUT") throw error;
    if (rows.length === 1) return [[rows[0][0], -1, 0, 0, 0, 0, 0, 0, 0, 0]];
    const middle = Math.floor(rows.length / 2);
    return processRows(rows.slice(0, middle)).concat(processRows(rows.slice(middle)));
  }
}

console.log(["row", "rule_union", "first_rule", "line_count", "first_line", "last_line", "distinct_lines", "min_gap", "valid_lines", "source_lines"].join(","));
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
