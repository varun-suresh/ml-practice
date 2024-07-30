# Transformers

("Attention is all you need")[https://arxiv.org/pdf/1706.03762] introduced this architecture that has been extremely successful in NLP, Vision and is behind apps like chatGPT. 

Here, I try to create some models trained on a small corpus of text data to better understand how these models work. I copied the two examples from (minGPT)[https://github.com/karpathy/minGPT]

### Example 1:

Sort Dataset

I want the model to learn how to sort a list of N (6) integers. To simplify the problem, the only integers I used are [0,1,2].

For example, if the input is [1,2,0,0,1,2], I want the output to be [0,0,1,1,2,2]

To train the transformer, I append the output(excluding the last number) to the input. For the example above, the input and the output to the transformer will be
```
[1,2,0,0,1,2,0,0,1,1,2]

[-1,-1,-1,-1,-1,0,0,1,1,2,2]
```
where -1 indicates an ignore character. 

### Example 2:

Character level Shakespeare Dataset

I want the model to generate text after being trained on all the works of Shakespeare (here)[https://github.com/varun-suresh/ml-practice/data/shakespeare.txt]. Sample output from the model
```
BRUTUS:
Not will I love thee every man hour.

MENENIUS:
I have lighted it body,
She is not at mine own being for our purpose,
Lest thou be quiet.

QUEEN MARGARET:
Who shall this, my lord, this gentleman.
How well you are left and strength friendly long,
When some of common sons; but it is meet a
batives for thy life-trust for thumb, surn, some two
bend move the lower heart I slaughter
In my breallog after the stern blame her.

DUKE VINCENTIO:
Why should he died is court at Christian;
This purpose reconciled by for self,
With Lady And that they do not chamber,
Ye show fair may lived at thy birth,
Or how 'tis held, my lord, it greet out.

ANTIGONUS:
This is the busier whooes they have been
Lield enter be no white both the child.

POMPEY:
Why, then you may be?  this new spring of it now.

ROMEO:
But, some motion; neither would speak to their soldiers;
So she deliver our dommission.
And goose cords will him as York then?
Have I not you deny, not to be doubted lamp, of my lady.

QUEEN:
We, speak:
```