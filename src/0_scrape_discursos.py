import asyncio
import aiohttp
import random
import pandas as pd
from tqdm import tqdm


async def fetch_page(session, dep_id, leg, pagina, itens, params_base, max_retries=1, base_delay=1, backoff_factor=2, pbar=None):
    url = f"https://dadosabertos.camara.leg.br/api/v2/deputados/{dep_id}/discursos"

    for attempt in range(1, max_retries + 1):
        try:
            params = params_base.copy()
            params.update({"itens": itens, "pagina": pagina})
            if leg:
                params["idLegislatura"] = str(leg)

            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("dados", [])

        except Exception as err:
            if attempt == max_retries:
                raise
            delay = base_delay * (backoff_factor ** (attempt - 1))
            jitter = random.uniform(0, 0.1 * delay)
            total_delay = delay + jitter
            tqdm.write(f"🔄 Retry {attempt}/{max_retries} for dep {dep_id}, leg {leg}, page {pagina}, itens {itens} after {total_delay:.1f}s. Error: {type(err).__name__}")
            await asyncio.sleep(total_delay)


async def fetch_deputado(session, dep_id, legislaturas, params_base, pbar, semaphore):
    all_data = []
    async with semaphore:
        for leg in legislaturas:
            pagina = 1
            itens = 100

            while True:
                try:
                    page_data = await fetch_page(session, dep_id, leg, pagina, itens, params_base, pbar=pbar)

                    if not page_data:
                        break

                    for record in page_data:
                        record["deputado_id"] = dep_id
                        record["idLegislatura"] = leg
                    all_data.extend(page_data)

                    pbar.update(len(page_data))
                    pbar.set_description(f"Deputy {dep_id} | Leg {leg or 'All'} | Page {pagina} | Items {len(page_data)}")

                    pagina += 1

                except Exception:
                    if itens > 1:
                        itens = max(1, itens // 2)
                        # if itens<3:
                            # tqdm.write(f"⚠️ Reducing itens to {itens} for dep {dep_id}, leg {leg}, page {pagina}")
                    else:
                        success = False
                        retries = 10
                        for retry in range(retries):
                            try:
                                page_data = await fetch_page(session, dep_id, leg, pagina, itens, params_base, pbar=pbar)
                                if not page_data:
                                    success = True
                                    break
                                for record in page_data:
                                    record["deputado_id"] = dep_id
                                    record["idLegislatura"] = leg
                                all_data.extend(page_data)
                                pbar.update(len(page_data))
                                pbar.set_description(f"Deputy {dep_id} | Leg {leg or 'All'} | Page {pagina} | Items {len(page_data)}")
                                pagina += 1
                                itens = 100
                                success = True
                                break
                            except Exception as e:
                                delay = 2 ** retry
                                if retry > 9:
                                    tqdm.write(f"⏳ Final retries {retry+1}/{retries} failed for dep {dep_id}, leg {leg}, page {pagina}. Waiting {delay}s. Error: {type(e).__name__}")
                                await asyncio.sleep(delay)
                        if not success:
                            tqdm.write(f"❌ Giving up on dep {dep_id}, leg {leg}, page {pagina} after {retries} retries at itens=1")
                            pagina +=1 
                            continue

    return all_data


async def fetch_deputados_discursos_async(unique_ids, data_inicio=None, data_fim=None, id_legislatura=None, max_concurrent=10):
    if not id_legislatura:
        legislaturas = [None]
    elif isinstance(id_legislatura, (list, tuple)):
        legislaturas = id_legislatura
    else:
        legislaturas = [id_legislatura]

    params_base = {"ordenarPor": "dataHoraInicio", "ordem": "DESC"}
    if data_inicio:
        params_base["dataInicio"] = data_inicio
    if data_fim:
        params_base["dataFim"] = data_fim

    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession(headers={"Accept": "application/json"}) as session:
        with tqdm(total=0, desc="Processing Deputies", unit="items") as pbar:
            tasks = [fetch_deputado(session, dep_id, legislaturas, params_base, pbar, semaphore) for dep_id in unique_ids]
            results = await asyncio.gather(*tasks)

    all_data = [item for sublist in results for item in sublist]
    if all_data:
        df = pd.json_normalize(all_data)
        return df
    return None


# Function to read unique IDs
def get_unique_deputado_ids(filepath='./data/raw/deputados_historico.csv'):
    df = pd.read_csv(filepath)
    unique_ids = df['id'].unique().tolist()
    return unique_ids


# Example usage
if __name__ == "__main__":
    unique_ids = get_unique_deputado_ids()
    print(f"Fetched {len(unique_ids)} unique deputy IDs.")

    df = asyncio.run(fetch_deputados_discursos_async(
        unique_ids,
        id_legislatura=[52, 53, 54, 55, 56],
        max_concurrent=30  # tweak concurrency here
    ))

    if df is not None:
        df.to_csv('./data/raw/deputados_discursos_async.csv', index=False)
        print(f"✅ Collected {len(df)} records")
    else:
        print("❌ No data fetched.")