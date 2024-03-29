# Fluxformer: Flow-Guided Duplex Attention Transformer via Spatio-Temporal Clustering for Action Recognition
Younggi Hong, Min Ju Kim, Isack Lee, Seok Bong Yoo*

(abstract)
Vision transformers have demonstrated impressive performance in various robotics and automation applications, such as classification automation and action recognition. However, the drawback of transformers is their quadratic increase in computing resources with larger inputs and dependence on considerable data for training. Most action recognition models using the transformer structure rely on a few frames from the original video to reduce computation, so temporal information is compromised by low frame rates. Spatial information is also compromised by reducing the number of embeddings as the transformer layer iterates. The paper proposes a robust model for action recognition that overcomes the limitations of most action recognition models with the transformer structure using the duplex attention function, flow-guided information, RGB information, and spatial support tokens. The proposed duplex attention mechanism leverages optical flow and RGB to address the lack of temporal information. The method employs spatial interest clustering to convert input data into tokens, improving the preservation of spatial information. Finally, meaningful action event frames are extracted by analyzing the flow and clustering to distinguish scenes. The experimental results reveal that the proposed model outperforms state-of-the-art methods in action recognition accuracy. Our source codes with pretrained models are available at https://github.com/YGspace/Fluxformer.




<img width="1674" alt="Overall" src="https://github.com/YGspace/Fluxformer/assets/86955204/9318b715-5f95-4ae1-bf21-ce1bd8bb7692">


