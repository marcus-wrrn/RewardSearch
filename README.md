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

The spymaster is the primary focus of the project, which is implemented via a GNN (Graph Neural Network). The model recieves the game state as its input and it must output an embedding which is then passed through a vector search algorithm to select the most similar word in its vocabulary. The current vocabulary is the entirety of the Merriam Webster dictionary, roughly 63,000 potential choices. 
This task can be thought of as a traditional similarity search method, with the addition that the model is searching for a final output that satisfies multiple objectives. This is referred to as Multi-Objective Retrieval in some parts of the code.

The loss function used in the model is unique, using a modified version of triplet loss alongside a method referred to as Reward/Reinforcement search. Unlike typical Reinforcement learning where a reward is applied directly to the loss function, instead triplet loss is used to score the model outputs overall 'direction' against its most simililar and disimilar outputs within a given search window. 
Please refer to the `multi_objective_models.py` file  within the `src/models` directory for the full implementation.

Currently the models best implementation can guess a word that on average results in a selection rate of 5.03/9 target words selected with a negative word selection rate (the first non-target word selected) of about 40% compared to neutral being around 60%, the assassin is never selected during the full tests. 
I am still experimenting with the model and tweaking the loss function so I will have more detailed results posted soon. However, it should be noted that these results already show better performance than a person (typically only getting about 2-3/9 words correct on the first turn) and the neutral vs. negative word selection rate shows that the current loss function `RewardSearchLoss` is capable of handling multiple objectives in a similarity search related context.

Another intersting observation is that is the model output (not the search output) is used for the word selection task the model is capable of achieving a 8.46/9 target words selected in the first turn alongside much higher neutral vs. negative selection rates.
