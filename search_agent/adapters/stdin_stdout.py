import sys
import json
import asyncio
from typing import Dict, Any

class StdinStdoutAdapter:
    def __init__(self, retrieval_engine):
        self.retrieval_engine = retrieval_engine

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        query = request.get("query")
        mode = request.get("mode", "preference")
        if not query:
            return {"error": "Missing 'query' in request"}
        
        results = await self.retrieval_engine.retrieve(query, mode)
        return {"results": results}

    async def run(self):
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)
