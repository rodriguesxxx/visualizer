# Plant.AI Visualizer Web

Site React para visualizar resultados de segmentação do modelo `yolov8n-seg.pt`.

O painel foi pensado para o estágio v1 do modelo, ainda treinado com configuração inicial:

```python
results = model.train(
    data=DATA,
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    project=V1_PATH,
    name=MODEL_NAME,
    save=True,
    exist_ok=True
)
```

## O que a tela cobre

- Comparação antes/depois da imagem.
- Sobreposição de máscaras e bounding boxes.
- Upload local de imagem para pré-visualização.
- Contagem das classes atuais: `folha` e `fruto`.
- Gráficos de contagem e curva demonstrativa de treino v1.
- Lista de instâncias com confiança e descritores CIELab.

## Rodando

Suba primeiro a API do modelo:

```bash
cd ../model
venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Depois rode o visualizer:

```bash
npm install
npm run dev
```

Por padrão o site consome `http://localhost:8000`. Para trocar:

```bash
VITE_MODEL_API_URL=http://localhost:8000 npm run dev
```

## Deploy no GitHub Pages

O projeto já inclui um workflow em `.github/workflows/deploy-visualizer.yml` para publicar o conteúdo de `visualizer/dist` no GitHub Pages.

Antes de rodar o workflow, configure no GitHub:

1. Em `Settings > Pages`, selecione `GitHub Actions` como source.
2. Em `Settings > Secrets and variables > Actions > Variables`, crie `VITE_MODEL_API_URL` com a URL pública do Hugging Face Space, por exemplo:

```text
https://seu-space.hf.space
```

O build usa `VITE_BASE_PATH=/plant.ai/`, que corresponde ao repositório `rodriguesxxx/plant.ai`. A página publicada ficará em:

```text
https://rodriguesxxx.github.io/plant.ai/
```

Para testar localmente com a mesma configuração do Pages:

```bash
VITE_MODEL_API_URL=https://seu-space.hf.space VITE_BASE_PATH=/plant.ai/ npm run build
npm run preview
```

No Hugging Face, a API precisa responder por HTTPS e permitir CORS para o domínio do Pages. A API FastAPI deste repositório já usa `CORSMiddleware` com origem liberada.

### Hugging Face privado e `HF_TOKEN`

Não coloque `HF_TOKEN` no GitHub Pages, no `.env` do Vite, nem em variável com prefixo `VITE_`. O GitHub Pages é um site estático: qualquer token usado pelo frontend precisa ir para o JavaScript público e pode ser visto no navegador.

Use uma destas opções:

1. Se o token é necessário para o Space baixar modelo/artefatos privados, configure o token no próprio Hugging Face Space:

```text
Space > Settings > Variables and secrets > New secret
Name: HF_TOKEN
Value: hf_...
```

Nesse caso, o `visualizer` continua usando apenas:

```text
VITE_MODEL_API_URL=https://seu-space.hf.space
```

2. Se o Space/API também é privado e exige autenticação para receber chamadas do navegador, não chame o Space diretamente pelo GitHub Pages. Crie um proxy backend público controlado por você, guarde `HF_TOKEN` nesse backend e aponte `VITE_MODEL_API_URL` para o proxy. O fluxo fica:

```text
GitHub Pages -> Proxy backend -> Hugging Face privado
```

O proxy precisa repassar para o Hugging Face:

```http
Authorization: Bearer $HF_TOKEN
```

## Estrutura de dados esperada

A tela consome `POST /api/v1/inference/analyze` e espera detecções no formato:

```ts
type Detection = {
  id: string;
  className: "folha" | "fruto";
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
  mask: string;
  area: number;
  lab: { l: number; a: number; b: number };
};
```

As coordenadas estão em porcentagem para manter a sobreposição responsiva. Quando a API estiver pronta, basta substituir os arrays `detections` e `trainingMetrics` pela resposta real.
