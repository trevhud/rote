/**
 * The run-record contract: one execution of a pipeline, as both the
 * hosted platform and OSS self-hosted runtimes record it.
 *
 * Field names deliberately match rote-cloud's D1 `runs` table (see its
 * `schema.sql` / `db.ts`): `wall_ms`, `input_tokens`, `output_tokens`,
 * `cost_usd`, and the `status` vocabulary. Anything consuming these
 * metrics reads the same shape regardless of where the run executed.
 */

/**
 * Lifecycle status of a run.
 *
 * `queued` / `running` / `paused` / `waiting` are in-flight; `complete`
 * is a settled success; `errored` / `terminated` are settled failures.
 * Speed/cost/determinism aggregations count `complete` runs only.
 */
export type RunStatus =
  | "queued"
  | "running"
  | "paused"
  | "waiting"
  | "complete"
  | "errored"
  | "terminated";

/** One recorded pipeline run. */
export interface RunRecord {
  /** Stable run id (also the workflow instance id in rote-cloud). */
  id?: string;
  status: RunStatus;
  /** The pipeline's result, or null when not (yet) settled successfully.
   *  Stored canonicalized in rote-cloud, but kept `unknown` here so the
   *  determinism metric can canonicalize on its own terms. */
  output: unknown | null;
  /** Failure detail for `errored` / `terminated` runs. */
  error?: string | null;
  /** Wall-clock duration in milliseconds, or null while unsettled. */
  wall_ms: number | null;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  started_at?: string;
  finished_at?: string | null;
}
