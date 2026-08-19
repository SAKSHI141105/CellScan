import { Card, CardContent, CardTitle, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AnimatedNumber } from "@/components/ui/animated-number";
import { riskTier } from "@/lib/utils";

export function PredictionResultCard({
  predictedClass,
  probability,
}: {
  predictedClass: string;
  probability: number;
}) {
  const { label, tier } = riskTier(probability);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap items-baseline gap-5">
          <div className="font-mono-num text-4xl font-bold text-foreground">
            <AnimatedNumber value={probability * 100} decimals={1} suffix="%" />
          </div>
          <div>
            <div className="text-base font-semibold">{predictedClass}</div>
            <Badge tier={tier} className="mt-1">
              {label}
            </Badge>
          </div>
        </div>
        <p className="mt-3 text-xs text-muted-foreground">probability of malignancy</p>
      </CardContent>
    </Card>
  );
}
