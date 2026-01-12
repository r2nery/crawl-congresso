import asyncio
import aiohttp
import pandas as pd
import glob
import os
import random
from tqdm import tqdm

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
# Pasta onde estão os arquivos proposicoesAutores-YYYY.csv
INPUT_PATTERN = './data/raw/proposicoes/proposicoesAutores-*.csv'
OUTPUT_PATH = './data/raw/proposicoes/proposicoes_detalhes_completo.parquet'
MAX_CONCURRENT = 20 # Ajustado para velocidade sem derrubar a API
BATCH_SIZE = 1000   # Salvar progresso a cada X registros se desejar

async def fetch_proposition_detail(session, prop_id, semaphore, pbar):
    """Busca detalhes de uma proposição específica."""
    url = f"https://dadosabertos.camara.leg.br/api/v2/proposicoes/{prop_id}"
    async with semaphore:
        for attempt in range(5):
            try:
                async with session.get(url, timeout=30) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(random.uniform(5, 15) * (attempt + 1))
                        continue
                    if resp.status == 404:
                        pbar.update(1)
                        return None
                    
                    resp.raise_for_status()
                    data = await resp.json()
                    dados = data.get('dados', {})
                    
                    pbar.update(1)
                    return {
                        'idProposicao': dados.get('id'),
                        'dataApresentacao': dados.get('dataApresentacao'),
                        'siglaTipo': dados.get('siglaTipo'),
                        'ementa': dados.get('ementa'),
                        'keywords': dados.get('keywords'),
                        'descricaoTipo': dados.get('descricaoTipo')
                    }
            except Exception:
                await asyncio.sleep(2 ** attempt)
        
        pbar.update(1)
        return None

async def main():
    # 1. Obter IDs Únicos de todas as proposições já baixadas
    print("--- Lendo arquivos de autores para extrair IDs únicos ---")
    files = glob.glob(INPUT_PATTERN)
    all_ids = set()
    for f in tqdm(files, desc="Lendo CSVs"):
        df_tmp = pd.read_csv(f, sep=';', usecols=['idProposicao'])
        all_ids.update(df_tmp['idProposicao'].unique().tolist())
    
    unique_ids = list(all_ids)
    print(f"Total de proposições únicas para raspar: {len(unique_ids)}")

    # 2. Verificar se já existe progresso salvo para não repetir
    if os.path.exists(OUTPUT_PATH):
        df_check = pd.read_parquet(OUTPUT_PATH, columns=['idProposicao'])
        ids_already_done = set(df_check['idProposicao'].unique())
        unique_ids = [pid for pid in unique_ids if pid not in ids_already_done]
        print(f"IDs restantes após verificar checkpoint: {len(unique_ids)}")

    if not unique_ids:
        print("Tudo já foi raspado!")
        return

    # 3. Execução Assíncrona
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async with aiohttp.ClientSession(headers={"Accept": "application/json"}) as session:
        with tqdm(total=len(unique_ids), desc="Raspando Detalhes") as pbar:
            tasks = [fetch_proposition_detail(session, pid, semaphore, pbar) for pid in unique_ids]
            results = await asyncio.gather(*tasks)

    # 4. Consolidar e Salvar
    new_data = [r for r in results if r is not None]
    if new_data:
        df_new = pd.DataFrame(new_data)
        
        # Se já existir arquivo, fazemos o append
        if os.path.exists(OUTPUT_PATH):
            df_old = pd.read_parquet(OUTPUT_PATH)
            df_final = pd.concat([df_old, df_new]).drop_duplicates(subset='idProposicao')
        else:
            df_final = df_new
            
        df_final.to_parquet(OUTPUT_PATH, index=False, compression='brotli')
        print(f"✅ Sucesso: {len(df_final)} proposições detalhadas salvas em {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())