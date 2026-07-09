/**
 * rote-metrics — the shared implementation of rote's empirical metrics
 * (determinism, speed, cost) plus the run-record and scorecard type
 * contracts. Zero runtime dependencies; consumed by rote-cloud and by
 * OSS self-hosted dashboards so both report identical numbers.
 */
export { canonicalize, flattenLeaves } from "./canonical.js";
export { computeDeterminism, type DeterminismReport } from "./determinism.js";
export { percentile, summarizeRuns, type RunSummary } from "./speed-cost.js";
export { type RunRecord, type RunStatus } from "./run-record.js";
export {
  type Range,
  type NodeKind,
  type ModelTier,
  type ScorecardNode,
  type ScorecardBefore,
  type ScorecardAfter,
  type CostRow,
  type ScorecardPriors,
  type Scorecard,
} from "./scorecard.js";
