/** @type {import('jest').Config} */
module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  roots: ["<rootDir>/e2e"],
  testMatch: ["**/*.e2e.ts"],
  testTimeout: 30_000,
  transform: {
    "^.+\\.ts$": ["ts-jest", { tsconfig: "tsconfig.e2e.json" }],
  },
};
