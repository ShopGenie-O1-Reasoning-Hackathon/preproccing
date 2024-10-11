# Fashion Product Embedding and Retrieval Project

[![Qdrant Collection Vector Points](https://fgdanyiprenrzvmxnjxw.supabase.co/storage/v1/object/public/statics/photo_2024-08-22_21-26-16.jpg)](https://huggingface.co/datasets/Melikshah/products)

## Project Overview

This project focuses on embedding and retrieval of a large-scale fashion product dataset collected from major brands such as Aarong, Allen Solly, Bata, Apex, and Infinity. The dataset consists of over 20,000 products, covering a wide variety of categories and styles. The notebook leverages powerful models and tools to create embeddings for both text and images, and then stores these embeddings in a vector database using Qdrant. This setup enables efficient and accurate retrieval of fashion products based on semantic similarity.

## Key Features

### Dataset Details
The dataset, hosted on Hugging Face, includes over 20,000 fashion products scraped from multiple sources, with details like product category, company, name, description, specifications, image links, and more. You can explore the dataset [here](https://huggingface.co/datasets/Melikshah/products).

### Embedding Models
- **Text Embeddings:** The notebook uses OpenAI's `text-embedding-3-large` model to create high-dimensional embeddings for the product descriptions and summaries.
- **Image Embeddings:** CLIP (`clip-ViT-B-32`) from the `SentenceTransformer` library is employed to generate image embeddings. This model captures visual features that can be used to find similar products based on their appearance.

### Embedding Strategy
For each product, a summary string is generated, capturing key details such as category, company, name, and specifications. This string is then embedded using the text model. Simultaneously, the primary product image is downloaded, processed, and encoded to generate an image embedding. Both embeddings are stored in the Qdrant collection for efficient vector search.

### Qdrant Vector Database
The Qdrant database is employed as a vector store for these embeddings, supporting real-time similarity searches based on both text and image queries. The notebook creates a collection that accommodates both summary and image vectors using cosine similarity.

### Scalable Data Pipeline
The notebook iterates over the dataset and:
1. Generates unique document IDs.
2. Prepares summary strings for text embedding.
3. Downloads and processes product images.
4. Computes embeddings for both text and images.
5. Stores the embeddings and related metadata (like product ID, links, and descriptions) into Qdrant.

This setup allows seamless integration into any system requiring fashion product recommendations or search functionality based on multi-modal data.

### Example of Qdrant Vector Points
The image above showcases the number of vector points stored in the Qdrant collection, visualizing the scale of the dataset and the embeddings stored.

## Usage Instructions
1. Clone the repository and install the necessary dependencies.
2. Load the dataset from Hugging Face.
3. Run the notebook to start embedding and storing fashion products in Qdrant.

The project is an excellent resource for anyone looking to explore multi-modal embeddings, vector databases, and fashion data at scale.

# LLaVA Product Description Generator

This project utilizes the LLaVA (Language and Vision Assistant) model to generate product descriptions and specifications from images. The model is based on a conversational AI architecture that can interact with both text and visual inputs.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.7+
- Google Colab or a local environment with GPU support
- Hugging Face's `transformers` and `datasets` libraries
- `torch` for PyTorch support
- `PIL` for image processing

### Installation

1. **Install the LLaVA package:**
    ```bash
    !pip install git+https://github.com/haotian-liu/LLaVA.git@786aa6a19ea10edc6f574ad2e16276974e9aaa3a
    ```

2. **Install additional dependencies:**
    ```bash
    !pip install -qU datasets
    ```

### Usage

1. **Initialize the LLaVA ChatBot:**
    ```python
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from llava.model import LlavaLlamaForCausalLM
    from llava.utils import disable_torch_init
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    from llava.conversation import conv_templates, SeparatorStyle
    import torch
    from PIL import Image
    import requests
    from io import BytesIO

    chatbot = LLaVAChatBot(load_in_8bit=True,
                           bnb_8bit_compute_dtype=torch.float16,
                           bnb_8bit_use_double_quant=True,
                           bnb_8bit_quant_type='nf8')
    ```

2. **Load the dataset:**
    ```python
    from datasets import load_dataset

    fashion = load_dataset(
        "thegreyhound/demo2",
        split="train"
    )
    product_df = fashion.to_pandas()
    ```

3. **Generate product descriptions and specifications:**
    ```python
    cnt = 1
    for index, row in product_df.iterrows():
        str1 = "Given Image detail was: " + row['Description'] + " Now generate a brief high level description for the product shown in the image"
        str2 = "Given Image detail was: " + row['Description'] + " Now generate a detailed specifications for the product shown in the image including the fabric, color, design, style etc"
        
        ans1 = chatbot.start_new_chat(img_path=row['Image_link'],
                                      prompt=str1)
        ans2 = chatbot.start_new_chat(img_path=row['Image_link'],
                                      prompt=str2)
        
        product_df.loc[index, 'Description'] = ans1
        product_df.loc[index, 'Specifications'] = ans2
        
        print(cnt)
        cnt += 1
    ```

### Example Output

The script processes images and generates high-level product descriptions and detailed specifications. The final output is saved in a JSON file containing an array of product information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LLaVA Project](https://github.com/haotian-liu/LLaVA)


## Author
- [Nazmus Sakib](https://www.linkedin.com/in/nazmus-sakib-touhid-a43533205).

You can find more details and access the dataset at [Hugging Face](https://huggingface.co/datasets/Melikshah/products).
