import { useEffect, useRef, useState } from "react";

const DURATION_MS = 600;
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export function AnimatedNumber({
  value,
  decimals = 1,
  suffix = "",
  className,
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  className?: string;
}) {
  const [display, setDisplay] = useState(0);
  const fromRef = useRef(0);

  useEffect(() => {
    const from = fromRef.current;
    const start = performance.now();
    let frame: number;

    function tick(now: number) {
      const progress = Math.min((now - start) / DURATION_MS, 1);
      setDisplay(from + (value - from) * easeOut(progress));
      if (progress < 1) frame = requestAnimationFrame(tick);
      else fromRef.current = value;
    }
    frame = requestAnimationFrame(tick);

    // requestAnimationFrame is throttled/suspended for backgrounded or
    // non-visible tabs, which would otherwise leave the number stuck
    // mid-tween — this guarantees the correct final value lands regardless.
    const settle = setTimeout(() => {
      fromRef.current = value;
      setDisplay(value);
    }, DURATION_MS + 50);

    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(settle);
    };
  }, [value]);

  return (
    <span className={className}>
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
}
