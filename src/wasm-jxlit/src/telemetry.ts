export interface Measure {
  name: string;
  startNs: number;
  durationNs: number;
}

export interface NativeDecodeTelemetry {
  rustTimebase: number;
  totalNs: number;
  measures: Measure[];
}

export interface DecodeTelemetry {
  timebase: number;
  totalNs: number;
  measures: Measure[];
}

export function unixTimeNs(): number {
  const millis = Date.now();
  return Math.trunc(millis * 1_000_000);
}

export function rebaseTelemetry(
  native: NativeDecodeTelemetry,
  timebase: number,
  wallNs: number,
  outerName: string,
): DecodeTelemetry {
  const delta = native.rustTimebase - timebase;
  const shifted = native.measures.map((measure) => ({
    name: measure.name,
    startNs: measure.startNs + delta,
    durationNs: measure.durationNs,
  }));
  return {
    timebase,
    totalNs: wallNs,
    measures: [{ name: outerName, startNs: 0, durationNs: wallNs }, ...shifted],
  };
}

export function telemetryToJson(telemetry: DecodeTelemetry): Record<string, unknown> {
  return {
    timebase: telemetry.timebase,
    total_ns: telemetry.totalNs,
    measures: telemetry.measures.map((measure) => ({
      name: measure.name,
      start_ns: measure.startNs,
      duration_ns: measure.durationNs,
    })),
  };
}

export function printPhaseSummary(
  telemetry: DecodeTelemetry,
  lang: string,
  topN = 10,
): void {
  const outer = telemetry.measures.find((measure) => measure.name.endsWith("_decode"));
  const totalNs = outer?.durationNs ?? telemetry.totalNs;
  const ranked = [...telemetry.measures]
    .sort((a, b) => b.durationNs - a.durationNs)
    .slice(0, topN);
  console.error(`\n== phase breakdown (${lang}) ==`);
  for (const measure of ranked) {
    const ms = measure.durationNs / 1_000_000;
    const pct = totalNs > 0 ? (100 * measure.durationNs) / totalNs : 0;
    console.error(`${measure.name.padEnd(16)} ${ms.toFixed(2).padStart(8)}ms ${pct.toFixed(1).padStart(6)}%`);
  }
}
