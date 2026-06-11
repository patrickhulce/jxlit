export interface Measure {
  name: string;
  startMs: number;
  durationMs: number;
}

export interface NativeDecodeTelemetry {
  rustTimebase: number;
  totalMs: number;
  measures: Measure[];
}

export interface DecodeTelemetry {
  timebase: number;
  totalMs: number;
  measures: Measure[];
}

export function unixTimeMs(): number {
  return Date.now();
}

export function rebaseTelemetry(
  native: NativeDecodeTelemetry,
  timebase: number,
  wallMs: number,
  outerName: string,
): DecodeTelemetry {
  const delta = native.rustTimebase - timebase;
  const shifted = native.measures.map((measure) => ({
    name: measure.name,
    startMs: measure.startMs + delta,
    durationMs: measure.durationMs,
  }));
  return {
    timebase,
    totalMs: wallMs,
    measures: [{ name: outerName, startMs: 0, durationMs: wallMs }, ...shifted],
  };
}

export function telemetryToJson(
  telemetry: DecodeTelemetry,
): Record<string, unknown> {
  return {
    timebase: telemetry.timebase,
    total_ms: telemetry.totalMs,
    measures: telemetry.measures.map((measure) => ({
      name: measure.name,
      start_ms: measure.startMs,
      duration_ms: measure.durationMs,
    })),
  };
}

export function printPhaseSummary(
  telemetry: DecodeTelemetry,
  lang: string,
  topN = 10,
): void {
  const outer = telemetry.measures.find((measure) =>
    measure.name.endsWith("_decode"),
  );
  const totalMs = outer?.durationMs ?? telemetry.totalMs;
  const ranked = [...telemetry.measures]
    .sort((a, b) => b.durationMs - a.durationMs)
    .slice(0, topN);
  console.error(`\n== phase breakdown (${lang}) ==`);
  for (const measure of ranked) {
    const pct = totalMs > 0 ? (100 * measure.durationMs) / totalMs : 0;
    console.error(
      `${measure.name.padEnd(16)} ${measure.durationMs.toFixed(2).padStart(8)}ms ${pct.toFixed(1).padStart(6)}%`,
    );
  }
}
