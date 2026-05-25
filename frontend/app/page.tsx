"use client";

import { useState } from "react";
import { Search, Sparkles, Loader2 } from "lucide-react";

type RecItem = {
  movie_idx: number;
  title: string | null;
  genres: string | null;
  score: number;
  rank: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8080";

export default function Home() {
  const [userIdx, setUserIdx] = useState(42);
  const [k, setK] = useState(10);
  const [diversity, setDiversity] = useState(0);
  const [items, setItems] = useState<RecItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function fetchRecs() {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_idx: userIdx, k, diversity_alpha: diversity }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setItems(data.items);
    } catch (e: any) {
      setError(e.message ?? "request failed");
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-900">
      <div className="max-w-4xl mx-auto p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-indigo-500" />
            TwoTowerRecs
          </h1>
          <p className="text-zinc-600 mt-1 text-sm">Two-tower retrieval on MovieLens</p>
        </header>

        <div className="bg-white rounded-lg shadow-sm border border-zinc-200 p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-zinc-700 mb-1">
                User index
              </label>
              <input
                type="number"
                value={userIdx}
                onChange={(e) => setUserIdx(parseInt(e.target.value || "0"))}
                className="w-full border border-zinc-300 rounded px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-400 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-700 mb-1">k</label>
              <input
                type="number"
                value={k}
                min={1}
                max={50}
                onChange={(e) => setK(parseInt(e.target.value || "10"))}
                className="w-full border border-zinc-300 rounded px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-400 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-700 mb-1">
                Diversity (MMR alpha)
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={diversity}
                onChange={(e) => setDiversity(parseFloat(e.target.value))}
                className="w-full"
              />
              <span className="text-xs text-zinc-500">{diversity.toFixed(2)}</span>
            </div>
            <div className="flex items-end">
              <button
                onClick={fetchRecs}
                disabled={loading}
                className="w-full inline-flex items-center justify-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded text-sm font-medium hover:bg-indigo-700 disabled:opacity-50"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                Recommend
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 rounded p-3 mb-4 text-sm">
            {error}
          </div>
        )}

        <div className="space-y-2">
          {items.map((it) => (
            <div
              key={it.movie_idx}
              className="bg-white border border-zinc-200 rounded-lg p-4 flex items-center gap-4"
            >
              <div className="text-2xl font-mono text-zinc-400 w-10 text-center">
                {it.rank + 1}
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-medium text-zinc-900 truncate">
                  {it.title ?? `Movie ${it.movie_idx}`}
                </div>
                {it.genres && (
                  <div className="text-xs text-zinc-500 mt-0.5">{it.genres}</div>
                )}
              </div>
              <div className="text-right">
                <div className="text-sm font-mono text-zinc-700">
                  {it.score.toFixed(3)}
                </div>
                <div className="text-xs text-zinc-400">score</div>
              </div>
            </div>
          ))}
          {!loading && items.length === 0 && !error && (
            <p className="text-center text-zinc-400 text-sm py-8">
              Enter a user index and click Recommend.
            </p>
          )}
        </div>
      </div>
    </main>
  );
}
