## Abstract

Information Retrieval (IR) systems are at the core of search engines, data mining and recently Retrieval Augmented Generation (RAG), which is used to provide context to generative AI for knowledge intensive tasks. However, modern deep learning methods used in IR struggle to adapt to objectives that they were not trained for. Existing IR models require fine-tuning to accommodate for downstream tasks. This requires access to large amounts of labeled data and can potentially harm the models ability to generalize. Additionally fine-tuning large models means that their encoded outputs cannot be reused for other systems. To solve these problems we propose Reward Search, a novel loss function and IR architecture capable of retrieving infor-
mation with multiple user defined objectives. We demonstrate the models capability by playing the language game Codenames which requires finding text from a knowledge base that satisfies multiple parameters. Reward Search takes 10 minutes to train on mid-range commercial hardware, is capable of handling a dynamic number of inputs and all trainable parameters can be stored in a 50MB file


## What is Codenames
Codenames is a party game developed by Czech Games, for a detailed explanation on how to play please refer to either (https://czechgames.com/files/rules/codenames-rules-en.pdf) or unofficially (http://codewordsgame.com/how-to-play)

For the purpose of this project the rules are explained as followed.

There are 25 words chosen at random. 9 of the words are target words, 9 are negative words, 6 are neutral and the last word is the assassin word. The goal of the game is to find all target words without selecting any negative words (resulting in your opponents getting a point) or the assassin word (immediate gameover). If you select any word that is not a target word then your turn is over.
All words are chosen sequentially. 

Each team has two roles, the spymaster and the operative. The spymaster knows which class each word belongs to while the operative does not. The goal of the spymaster is to come up with a single word to give to the operative. The operative must then select all words on the board that they believe are most similar to their given word.
If they pick a word that is not one of their target words their turn ends. 

Thus, the spymaster has multiple objectives. They need to pick a word that will not only result in the highest amount of target words picked but that word should not be similar enough to their opponents words or the assassin word. Preferrably if the operative does pick a word it should be a neutral word since it does not give the other team a point or result in them losing the game.

### AI implementation
The operative role is implemented entirely using cosine similarity scores. The higher the similarity score between the final output is to each word in the environment, the higher its order in the selection process. The most similar word to the output is picked first, the second most similar word is picked second etc. Word embeddings are created via a sentence transformer, which are used for the selection process.

The spymaster is the primary focus of the project, which is implemented via a feedforward network with a Sentence Transformer backbone, alongside the HNSW vector search algorithm. The model recieves the game state as its input, which is represented as the mean pooled and normalized embeddings of all word classes, and outputs an embedding which corresponds to a word in a predefined vocabulary. The current vocabulary is the entirety of the Merriam Webster dictionary, roughly 63,000 potential choices.
 
This task can be thought of as a traditional similarity search method, with the addition that the model is searching for a final output that satisfies multiple objectives. This is referred to as Multi-Objective Retrieval. 

The loss function used in the model is unique, using a modified version of triplet loss alongside a method referred to as Reward/Reinforcement search. Unlike typical Reinforcement learning where a reward is applied directly to the loss function, instead triplet loss is used to score the model outputs overall 'rotation' against its highest and lowest scoring search results, in a given search window.
Please refer to the `multi_objective_models.py` file  within the `src/models` directory for the full implementation. 

### Long Text Support
The model has been shown to work on longer texts and in some cases performs better than single words. However, the long-text data pipeline is still being worked on and is not as streamlined as the single word model. 

### Frontend
This code is used purely for training and testing the RewardSearch model. For a working fronted please refer to the project `https://github.com/aclarke500/qmind-codenames-front-end` which is used to play against the model. Models trained here can be imported to the frontend to play against.

### Further Information
We have provided the most recent draft of our paper within the project. We are currently working on getting it put on arxiv and will be eventually released as part of the CUCAI conference proceedings. For any additional questions feel free to email `18mgw1@queensu.ca` about the project. 


