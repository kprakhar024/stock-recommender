"use client";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function TechnicalChart({ data }: { data: any[] }) {
  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#666" }} interval={15} />
          <YAxis tick={{ fontSize: 10, fill: "#666" }} domain={['auto', 'auto']} />
          <Tooltip contentStyle={{ background: "#111", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px" }} />
          <Line type="monotone" dataKey="close" stroke="#8b5cf6" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="sma20" stroke="#10b981" strokeWidth={1} dot={false} strokeDasharray="3 3" />
          <Line type="monotone" dataKey="sma50" stroke="#f59e0b" strokeWidth={1} dot={false} strokeDasharray="3 3" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
