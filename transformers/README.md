# Transformers

("Attention is all you need")[https://arxiv.org/pdf/1706.03762] introduced this architecture that has been extremely successful in NLP, Vision and is behind apps like chatGPT. 

Here, I try to create some models trained on a small corpus of text data to better understand how these models work. I copied the two examples from (minGPT)[https://github.com/karpathy/minGPT]

Example 1:

Sort Dataset

I want the model to learn how to sort a list of N (6) integers. To simplify the problem, the only integers I used are [0,1,2].

For example, if the input is [1,2,0,0,1,2], I want the output to be [0,0,1,1,2,2]

To train the transformer, I append the output(excluding the last number) to the input. For the example above, the input and the output to the transformer will be

[1,2,0,0,1,2,0,0,1,1,2]

[-1,-1,-1,-1,-1,-1,0,0,1,1,2,1]

where -1 indicates an ignore character. 

Example 2:

Character level Shakespeare Dataset

I want the model to generate text after being trained on all the works of Shakespeare (here)[data/shakespeare.txt]