# Raspberry AI - Take Home Exercise (AI role)
## By Adithya Sampath

## The two notebooks (`RaspberryAI_solution1.ipynb` and `RaspberryAI_solution2.ipynb`) cover 3 proposed solutions. The 3 approaches are as follows:

## 1. Using negative and positive tokens with stable diffusion + Clothing segmentation (Refer: `RaspberryAI_solution1.ipynb`)
### Step 1
Use learned embeddings (i.e. positive and negative tokens) for the CLIP text encoder. Midjourney are Wrong embeddings are used to update the CLIP tokenizer and text encoder to learn positive and negative tokens respectively. They are used with the stable diffusion pipeline to generate images in a certain style (i.e. Midjourney), and improve emphasis of negative prompts (i.e. Wrong) respectively. Using manual seed Generator help make the generation more deterministic, and different seeds make a significant impact in quality of results. `Seed = 19687` is used here. Example results of modified stable diffusion:

| Prompt | Result |
|---|---|
|  black short sleeve crewneck crop top  |  <img src="solution1\result_img_6.png" alt="input" width="300" />  |
|  black long sleeve crewneck crop top  |  <img src="solution1\result_img_5.png" alt="input" width="300" />  |

### Step 2
Once we obtain results from the stable diffusion model, we can use a clothing segmentation model to segment clothing articles in the generated image. Segformer B2 fine-tuned for clothes segmentation is used to detect 18 different classes. `class_id = 4` is the class for `Upper-clothes`, hence, is used to get a segmentation mask of the crop top. Similarly, `class id = 5` can be used for skirts, `class id = 6` for pants, `class id = 7` for dress, and so on. Once we have the segmentation mask, we can crop the crop top from the generated images, and paste in on a plain white background image. Results are as follows:

| Prompt | Result |
|---|---|
|  black short sleeve crewneck crop top  |  <img src="solution1\result_img6_out.png" alt="input" width="300" />  |
|  black long sleeve crewneck crop top  |  <img src="solution1\result_img5_out.png" alt="input" width="300" />  |

## 2. Text conditioned Image to Image Stable Diffusion (Refer: `RaspberryAI_solution2.ipynb`)

Assuming a large catalog of fashion images, use a Image captioning (ex. BLIP) model to generate image captions. Use CLIP Retrieval library to generate and index embeddings for images & text captions. Use query text to retrieve the most similar image in catalog using 1-NN KNN service. Use a text conditioned image to image stable diffusion model (i.e. Stable Diffusion InstructPix2Pix) to generate new images using 1-NN image and text query guidance.

| Prompt | Result |
|---|---|
|  black short sleeve crewneck crop top  |  <img src="solution2\black_short_sleeve_crop_top.png" alt="input" width="300" />  |

## 3. Stable Diffusion fine-tuned for Fashion Product Images Dataset (Refer: `RaspberryAI_solution2.ipynb`)
By fine-tuning stable diffusion on Fashion Product Images, we can produce more relevant fashion images.

| Prompt | Result |
|---|---|
|  black short sleeve crewneck crop top  |  <img src="solution2\black_croptop.png" alt="input" width="300" />  |