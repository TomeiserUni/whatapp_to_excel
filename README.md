# WhatsApp to Excel

Aplicacao Flask para transformar pedidos enviados por WhatsApp em linhas de encomenda prontas para exportar para Excel.

O projeto reconhece produtos a partir de texto colado ou imagens, cruza o resultado com o catalogo local, aplica aliases/regras de matching e gera um ficheiro `resultados.xlsx` com referencia, produto, quantidade e origem do reconhecimento.

## Funcionalidades

- Processamento de texto colado de mensagens WhatsApp.
- Processamento de imagens com OCR.
- Matching de produtos por aliases, regras deterministicas, embeddings e, opcionalmente, Claude.
- Exportacao dos resultados para Excel.
- Login opcional para controlar o uso de tokens.
- Registo de linhas nao reconhecidas para ajudar a melhorar os aliases.
- Sincronizacao do catalogo via Shopkit API.

## Estrutura

```text
src/
  server.py          Servidor Flask e rotas da aplicacao
  pipeline.py        Pipeline local com OCR, embeddings e regras
  ai_pipeline.py     Pipeline com Claude para texto/imagem
  shopkit_api.py     Atualiza catalogo a partir da API Shopkit
  templates/         Paginas HTML da interface
data/
  prod.pkl           Lista de produtos
  sku_map.pkl        Mapa produto -> referencia/utilizar para fazer chamadas caso o servidor estiver lento
  aliases.json       Aliases manuais para melhorar o matching
  exemplos.json      Exemplos usados pelo pipeline de AI
  emb_prod.npy       Embeddings dos produtos
input/               Pasta local para ficheiros de entrada
output/              Pasta local para ficheiros gerados
```

## Requisitos

- Python 3.11 recomendado
- `pip`
- Chave Anthropic opcional para usar Claude
- Chave Shopkit opcional para atualizar o catalogo

## Instalacao Local

Cria e ativa um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instala as dependencias:

```bash
pip install -r requirements.txt
```

Cria um ficheiro `.env` na raiz do projeto, se ainda nao existir:

```env
ANTHROPIC_API_KEY=sua_chave_anthropic
SHOPKIT_API_KEY=sua_chave_shopkit
SECRET_KEY=uma_chave_secreta_para_sessao
APP_USERNAME=admin
APP_PASSWORD=uma_password_segura
PORT=5002
```

Notas:

- `ANTHROPIC_API_KEY` e opcional localmente, mas melhora o processamento de texto.
- `APP_USERNAME` e `APP_PASSWORD` ativam login quando `APP_PASSWORD` estiver definida.
- No macOS/Linux o porto padrao e `5002`; no Windows e `5000`.

## Como Correr

```bash
python src/server.py
```

Depois abre:

```text
http://localhost:5002
```

Se definires outro `PORT` no `.env`, usa esse porto.

## Uso

1. Cola o texto da encomenda ou envia imagens do WhatsApp.
2. Clica para processar.
3. Revê os produtos, quantidades, referencias e scores.
4. Exporta para Excel.

O Excel gerado inclui cores por confianca:

- Verde: correspondencia forte.
- Amarelo: correspondencia aceitavel, convem rever.
- Vermelho: produto nao identificado ou score baixo.

## Melhorar Reconhecimento

Edita `data/aliases.json` para adicionar nomes comuns, abreviaturas ou erros frequentes:

```json
{
  "acrigel 5": "polyacrygel 5",
  "base rubber": "rubber base"
}
```

As linhas nao reconhecidas sao registadas em:

```text
data/nao_reconhecidas.jsonl
```

Tambem podes consultar tendencias pela rota:

```text
/tendencias
```

## Atualizar Catalogo Shopkit

Com `SHOPKIT_API_KEY` definido no `.env`:

```bash
python src/shopkit_api.py
```

Este comando atualiza:

- `data/prod.pkl`
- `data/sku_map.pkl`
- `data/prod_meta.json`
- `data/emb_prod.npy`

## Docker / Cloud

Para construir a imagem:

```bash
docker build -t whatsapp-to-excel .
```

Para correr:

```bash
docker run --env-file .env -p 5002:5002 whatsapp-to-excel
```

Em cloud, define pelo menos:

```env
ANTHROPIC_API_KEY=sua_chave_anthropic
SECRET_KEY=uma_chave_secreta
APP_USERNAME=admin
APP_PASSWORD=uma_password_segura
PORT=5002
```

O projeto deteta ambientes como Railway/Render por `RAILWAY_ENVIRONMENT` ou `RENDER`.

## Desenvolvimento

Ficheiros mais importantes:

- `src/server.py`: rotas, autenticacao, preprocessamento e exportacao.
- `src/pipeline.py`: OCR local, embeddings e matching.
- `src/ai_pipeline.py`: matching com Claude.
- `data/aliases.json`: principal ficheiro para afinar resultados rapidamente.
- `src/templates/index.html`: interface principal.

Depois de alterar aliases ou dados, reinicia o servidor para garantir que tudo e recarregado.
