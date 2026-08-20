import { Card, CardContent, CardTitle, CardHeader } from "@/components/ui/card";
import { RiskGauge } from "@/components/ui/risk-gauge";

export function PredictionResultCard({
  predictedClass,
  probability,
}: {
  predictedClass: string;
  probability: number;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center gap-1 pt-2">
        <RiskGauge probability={probability} />
        <p className="text-sm font-semibold">{predictedClass}</p>
        <p className="text-xs text-muted-foreground">probability of malignancy</p>
      </CardContent>
    </Card>
  );
}
