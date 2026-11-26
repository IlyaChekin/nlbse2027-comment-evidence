// Activate the version of the conversion rule limited to finite numeric literals.
process.argv.push("--bounded-numeric");
await import("./scan.mjs");
