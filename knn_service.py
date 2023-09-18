# NOTE: TASK 3

from clip_retrieval.clip_back import *

class KnnServer:
    """the knn service provides nearest neighbors given text or image"""

    def __init__(self, clip_resources):
        self.clip_resources = clip_resources

    def compute_query(
        self,
        clip_resource,
        text_input,
        embedding_input=None,
        image_input=None,
        image_url_input=None,
        aesthetic_score=None,
        aesthetic_weight=None,
        use_mclip=False
    ):
        """compute the query embedding"""
        import torch  # pylint: disable=import-outside-toplevel

        if text_input is not None and text_input != "":
            if use_mclip:
                with TEXT_CLIP_INFERENCE_TIME.time():
                    query = normalized(clip_resource.model_txt_mclip(text_input))
            else:
                with TEXT_PREPRO_TIME.time():
                    text = clip_resource.tokenizer([text_input]).to(clip_resource.device)
                with TEXT_CLIP_INFERENCE_TIME.time():
                    with torch.no_grad():
                        text_features = clip_resource.model.encode_text(text)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    query = text_features.cpu().to(torch.float32).detach().numpy()
        elif image_input is not None or image_url_input is not None:
            if image_input is not None:
                binary_data = base64.b64decode(image_input)
                img_data = BytesIO(binary_data)
            elif image_url_input is not None:
                img_data = download_image(image_url_input)
            with IMAGE_PREPRO_TIME.time():
                img = Image.open(img_data)
                prepro = clip_resource.preprocess(img).unsqueeze(0).to(clip_resource.device)
            with IMAGE_CLIP_INFERENCE_TIME.time():
                with torch.no_grad():
                    image_features = clip_resource.model.encode_image(prepro)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                query = image_features.cpu().to(torch.float32).detach().numpy()
        elif embedding_input is not None:
            query = np.expand_dims(np.array(embedding_input).astype("float32"), 0)

        if clip_resource.aesthetic_embeddings is not None and aesthetic_score is not None:
            aesthetic_embedding = clip_resource.aesthetic_embeddings[aesthetic_score]
            query = query + aesthetic_embedding * aesthetic_weight
            query = query / np.linalg.norm(query)

        return query


    def connected_components(self, neighbors):
        """find connected components in the graph"""
        seen = set()

        def component(node):
            r = []
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= set(neighbors[node]) - seen
                r.append(node)
            return r

        u = []
        for node in neighbors:
            if node not in seen:
                u.append(component(node))
        return u

    def get_non_uniques(self, embeddings, threshold=0.94):
        """find non-unique embeddings"""
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)  # pylint: disable=no-value-for-parameter
        l, _, I = index.range_search(embeddings, threshold)  # pylint: disable=no-value-for-parameter,invalid-name

        same_mapping = defaultdict(list)

        # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
        for i in range(embeddings.shape[0]):
            for j in I[l[i] : l[i + 1]]:
                same_mapping[int(i)].append(int(j))

        groups = self.connected_components(same_mapping)
        non_uniques = set()
        for g in groups:
            for e in g[1:]:
                non_uniques.add(e)

        return list(non_uniques)

    def connected_components_dedup(self, embeddings):
        non_uniques = self.get_non_uniques(embeddings)
        return non_uniques

    def get_unsafe_items(self, safety_model, embeddings, threshold=0.5):
        """find unsafe embeddings"""
        nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
        x = np.array([e[0] for e in nsfw_values])
        return np.where(x > threshold)[0]

    def get_violent_items(self, safety_prompts, embeddings):
        safety_predictions = np.einsum("ij,kj->ik", embeddings, safety_prompts)
        safety_results = np.argmax(safety_predictions, axis=1)
        return np.where(safety_results == 1)[0]

    def post_filter(
        self, safety_model, embeddings, deduplicate, use_safety_model, use_violence_detector, violence_detector
    ):
        """post filter results : dedup, safety, violence"""
        to_remove = set()
        if deduplicate:
            with DEDUP_TIME.time():
                to_remove = set(self.connected_components_dedup(embeddings))

        if use_violence_detector and violence_detector is not None:
            to_remove |= set(self.get_violent_items(violence_detector, embeddings))
        if use_safety_model and safety_model is not None:
            with SAFETY_TIME.time():
                to_remove |= set(self.get_unsafe_items(safety_model, embeddings))

        return to_remove

    def knn_search(
        self, query, modality, num_result_ids, clip_resource, deduplicate, use_safety_model, use_violence_detector
    ):
        """compute the knn search"""

        image_index = clip_resource.image_index
        text_index = clip_resource.text_index
        if clip_resource.metadata_is_ordered_by_ivf:
            ivf_old_to_new_mapping = clip_resource.ivf_old_to_new_mapping

        index = image_index if modality == "image" else text_index

        with KNN_INDEX_TIME.time():
            if clip_resource.metadata_is_ordered_by_ivf:
                previous_nprobe = faiss.extract_index_ivf(index).nprobe
                if num_result_ids >= 100000:
                    nprobe = math.ceil(num_result_ids / 3000)
                    params = faiss.ParameterSpace()
                    params.set_index_parameters(index, f"nprobe={nprobe},efSearch={nprobe*2},ht={2048}")
            distances, indices, embeddings = index.search_and_reconstruct(query, num_result_ids)
            if clip_resource.metadata_is_ordered_by_ivf:
                results = np.take(ivf_old_to_new_mapping, indices[0])
            else:
                results = indices[0]
            if clip_resource.metadata_is_ordered_by_ivf:
                params = faiss.ParameterSpace()
                params.set_index_parameters(index, f"nprobe={previous_nprobe},efSearch={previous_nprobe*2},ht={2048}")
        nb_results = np.where(results == -1)[0]

        if len(nb_results) > 0:
            nb_results = nb_results[0]
        else:
            nb_results = len(results)
        result_indices = results[:nb_results]
        result_distances = distances[0][:nb_results]
        result_embeddings = embeddings[0][:nb_results]
        result_embeddings = normalized(result_embeddings)
        local_indices_to_remove = self.post_filter(
            clip_resource.safety_model,
            result_embeddings,
            deduplicate,
            use_safety_model,
            use_violence_detector,
            clip_resource.violence_detector,
        )
        indices_to_remove = set()
        for local_index in local_indices_to_remove:
            indices_to_remove.add(result_indices[local_index])
        indices = []
        distances = []
        for ind, distance in zip(result_indices, result_distances):
            if ind not in indices_to_remove:
                indices_to_remove.add(ind)
                indices.append(ind)
                distances.append(distance)

        return distances, indices

    def map_to_metadata(self, indices, distances, num_images, metadata_provider, columns_to_return):
        """map the indices to the metadata"""

        results = []
        with METADATA_GET_TIME.time():
            metas = metadata_provider.get(indices[:num_images], columns_to_return)
        for key, (d, i) in enumerate(zip(distances, indices)):
            output = {}
            meta = None if key + 1 > len(metas) else metas[key]
            convert_metadata_to_base64(meta)
            if meta is not None:
                output.update(meta_to_dict(meta))
            output["id"] = i.item()
            output["similarity"] = d.item()
            results.append(output)

        return results

    def query(
        self,
        text_input,
        embedding_input=None,
        modality="image",
        num_images=10,
        num_result_ids=10,
        aesthetic_score=None,
        aesthetic_weight=None,
        image_input=None,
        image_url_input=None,
        indice_name=None,
        use_mclip=False,
        deduplicate=True,
        use_safety_model=False,
        use_violence_detector=False,
    ):
        """implement the querying functionality of the knn service: from text and image to nearest neighbors"""

        if text_input is None and image_input is None and image_url_input is None and embedding_input is None:
            raise ValueError("must fill one of text, image and image url input")
        if indice_name is None:
            indice_name = next(iter(self.clip_resources.keys()))

        clip_resource = self.clip_resources[indice_name]

        query = self.compute_query(
            clip_resource=clip_resource,
            text_input=text_input,
            image_input=image_input,
            image_url_input=image_url_input,
            embedding_input=embedding_input,
            use_mclip=use_mclip,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,
        )
        distances, indices = self.knn_search(
            query,
            modality=modality,
            num_result_ids=num_result_ids,
            clip_resource=clip_resource,
            deduplicate=deduplicate,
            use_safety_model=use_safety_model,
            use_violence_detector=use_violence_detector,
        )
        if len(distances) == 0:
            return []
        results = self.map_to_metadata(
            indices, distances, num_images, clip_resource.metadata_provider, clip_resource.columns_to_return
        )

        return results