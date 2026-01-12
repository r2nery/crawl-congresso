import asyncio
import aiohttp
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
DATA_INICIO = "2003-02-01" 
DATA_FIM = "2024-12-31"
MAX_CONCURRENT = 40 
OUTPUT_PATH = './data/raw/votacoes_detalhadas.parquet'

async def fetch_api_data(session, url, params=None, max_retries=5):
    """Fetch resiliente para a API da Câmara."""
    for attempt in range(1, max_retries + 1):
        try:
            async with session.get(url, params=params, timeout=60) as resp:
                if resp.status == 429: # Rate limit
                    await asyncio.sleep(5 * attempt)
                    continue
                if resp.status >= 400:
                    return []
                
                resp.raise_for_status()
                data = await resp.json()
                return data.get("dados", [])
        except Exception:
            if attempt == max_retries: return []
            await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
    return []

async def fetch_voting_event(session, vote_id, vote_date, pbar, semaphore):
    """Obtém orientações dos âncoras e a lista COMPLETA de votos."""
    all_votes_event = []
    async with semaphore:
        try:
            # 1. Buscar Todas as Orientações
            orient_url = f"https://dadosabertos.camara.leg.br/api/v2/votacoes/{vote_id}/orientacoes"
            orient_data = await fetch_api_data(session, orient_url)
            
            # Criamos um dicionário de busca rápida
            orient_dict = {o['siglaPartidoBloco']: o['orientacaoVoto'] for o in orient_data}
            
            # Extraímos os âncoras estáveis
            gov_orient = orient_dict.get('Governo', 'N/A')
            maioria_orient = orient_dict.get('Maioria', 'N/A')
            minoria_orient = orient_dict.get('Minoria', 'N/A')
            oposicao_orient = orient_dict.get('Oposição', 'N/A')

            # 2. Buscar Votos (Roll Call)
            votos_url = f"https://dadosabertos.camara.leg.br/api/v2/votacoes/{vote_id}/votos"
            votos_data = await fetch_api_data(session, votos_url)
            
            for v in votos_data:
                dep_info = v.get('deputado_', {})
                dep_id = dep_info.get('id') if dep_info else None
                dep_party = dep_info.get('siglaPartido') if dep_info else None
                
                all_votes_event.append({
                    'idVotacao': vote_id,
                    'dataVotacao': vote_date,
                    'deputado_id': dep_id,
                    'siglaPartido': dep_party,
                    'voto': v.get('tipoVoto'),
                    'gov_orient': gov_orient,
                    'maioria_orient': maioria_orient,
                    'minoria_orient': minoria_orient,
                    'oposicao_orient': oposicao_orient,
                    'blocos_orient': orient_dict # Dicionário completo para auditoria
                })

            pbar.update(1)
            pbar.set_description(f"Votação: {vote_id}")
        except Exception:
            pass
            
    return all_votes_event

async def fetch_all_votings_async():
    start_dt = datetime.strptime(DATA_INICIO, '%Y-%m-%d')
    end_dt = datetime.strptime(DATA_FIM, '%Y-%m-%d')
    
    chunks = []
    curr = start_dt
    while curr < end_dt:
        nxt = min(curr + timedelta(days=30), end_dt)
        chunks.append((curr.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')))
        curr = nxt + timedelta(days=1)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    all_voting_ids = []

    async with aiohttp.ClientSession(headers={"Accept": "application/json"}) as session:
        # PASSO 1: Listagem de IDs
        print(f"--- Listando IDs de votação ({len(chunks)} blocos) ---")
        for c_start, c_end in chunks:
            params = {"dataInicio": c_start, "dataFim": c_end, "ordem": "ASC", "ordenarPor": "data"}
            ids_chunk = await fetch_api_data(session, "https://dadosabertos.camara.leg.br/api/v2/votacoes", params)
            if ids_chunk:
                all_voting_ids.extend(ids_chunk)
        
        print(f"Total de votações para processar: {len(all_voting_ids)}")

        # PASSO 2: Detalhes dos Votos
        with tqdm(total=len(all_voting_ids), desc="Baixando Roll Calls") as pbar:
            tasks = [fetch_voting_event(session, v['id'], v['data'], pbar, semaphore) for v in all_voting_ids]
            results = await asyncio.gather(*tasks)

    flat_results = [item for sublist in results for item in sublist]
    if flat_results:
        df = pd.DataFrame(flat_results)
        # Usamos pyarrow para garantir suporte ao dicionário na coluna blocos_orient
        df.to_parquet(OUTPUT_PATH, index=False, compression='brotli', engine='pyarrow')
        print(f"✅ Sucesso: {len(df)} registros de votos salvos em {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(fetch_all_votings_async())