# These are only ideas. Don't execute anything from here

- [ ] save in a memory the way to create the dataset ready for training
- [ ] ~~Hay errors con cot: Yes 0133bcec_base_n1 Digit candidates not yet used are not correctly displayed~~
- [x] why there are examples with 1 example equation that doesn't have any possible perturbation??
- [x] Remove the concat doesn't makes sense. Add again
- [ ] ~~Maybe remove the synthetic data if the code present different solution the learning got confused about what mapping is the correct.~~ 
- [x] Reduce the number of perturbation maximum 2 (or just one) maximum chars allowed: 18,844 chars
- [ ] Añadir la cot al set de evaluacion para comparar uno a uno
- [x] Hay alguna manera de escoger el mapeo mas problema desde el comienzo (la permutacion de digitos mas probable) Basado en la estructura?? Digamos la estructura muesta ciertos patrones que imposibilita algunos digitos para ciertos symbolos. Hay alguna regla para reducirlos?? Incluso cambiar el order con el que se assignan
- [x] Dejar que calcule los offset siempre, no ser tan restrictivo y permitir -2 a 2 para 
- [ ] ~~Hay alguna formula para gdc y lcm~~
- [x] Rework of the operations (some values are not allowed) only wording on the first definitions
- [x] Rework the wording as finding a solution but if after some N tries is not gotten return the best guess that will my program expected.
- [x] How is handling situations that requires zero-padding or appending the sign at the end of the answer. That has to also be mentioned.

- [ ] Injectar matematicas basicas del dataset que vi hace poco (se esta equivocando)
- [ ] the cot adds a lot of mathematical machinery that can be difficult to understand. Injected some made up examples only on some parts during learning could help make the llm undertand a process like the table for choosing the intial digits
basic maths operation, etc, etc -> Try in a future version.
    - IMPORTANT: now, I think if it fails tomorrow.

- NOTE: The digit picking strategy that work in the first mapping could work in subsequent mappings, but now how is today because I'm forcing the change


- Equation Numeric problems look similar to the cryptharithm but the cot is totally different. I would like
    - Rewrite it similar to my style only rewriting
    - Create or augment data if possible also for the other cases where there are not enough data to keep balanced
    




### Check the following
- [ ] Add the chain of though for pure concat too
- [ ] Add randomly the some samples of standard and balance
- [ ] A rule per operation to check the structure based on a set of equation or only one equation.
- [ ] A rule to check operations that contradicts each other and get symplified
- [ ] Check symbols in equation and available operations ranks -> null some digits
- [ ] Check how many solution require the 0 padding

https://www.kaggle.com/models/kienngx/nemotron-nano-30b-trained/triton/tinker-adapter


## NOTES to try
- [x] review if the concatenation can also happen with less than 4 digits at the rhs
    - we have always for digits in the lhs (ojo)
- [ ] are there other factual observation that can be done using the structure of the equation for gcd, lcm, fdiv, mod, rmod, add, mul??. Pick 10 samples and check if there is room for other
- [ ] othe redudant observation. Pick 10 samples and check if there is room for other
- [ ] other follow up observation
- [x] add the approximate token spent on the cot

- When find a valid use VALID as something to stand up  that this is a possible valid solution

- It can be ok also to check all the offset every time.

### Ideas
- [ ] Shift the symbols in order according the assci order until the * - or + are in the middle and check what are the mappings
- [ ] Check if I can reduce the and correct those case. It looks like almost always * should be a kind of multiplication, + a concatatenation or sum and - a way of difference.
    - Use the same solver only why these examples and removing some operations that consider it shouldn't search on.
- I think it will be possible to add some kind of memoization (cache)


- If the operator appears at the last position in any of the equations. It is always a little endian strategy.
    - And notice is a signal to place the op symbol at the end in the solution if required.
- Most of 

- Solo calcula el error con la solucion, los token generados, no los token de chain of thought. In theory it could generate any change of thought. CONSULTAR SI TIENE SENTIDO Y COMO SE HACE?
- Utilizar con mas probabilidad ciertos simbolos para el simbolo del operador.


- Always check how far the search went with 0, 1, 2, 3 examples
- Additionally, removing constraints can make the database more random and cover more cases


- Use the percentages to create samples without any constrain that explain the operation of the result
- If necesary add a way to randonize the digits mapping for problem with less examples that could match a lot of cases.

- There is even a probability of how long is the answer

- [ ] Shift the symbols in order according the assci order until the * - or + are in the middle and check what are the mappings
- [ ] Check if I can reduce the and correct those case. It looks like almost always * should be a kind of multiplication, + a concatatenation or sum and - a way of difference.
    - Use the same solver only why these examples and removing some operations that consider it shouldn't search on.
- I think it will be possible to add some kind of memoization (cache)

- Don't expand the candidates. It's better to present all the mappings and just let the llm to take one of them to find the solution.
- Only the mappings and strategies are different


- Calculate with 0 examples all the response and maybe add to the prompt the first explanation section for the next datasets with 1, 2, and 3 examples.

- The zero padding is added based on the examples.
- Concat is an obvious clue always
    - it could be something like directly forward and rev
    - it could first apply little endian forward and rev


- Train 22K example 0
- Train 22k example 1 = 5k example 2
- Train 22k 

TODO:
- Fix the CoT for dataset with 1-5 examples
- Need other way to balance when there are more examples. Because now I just balancing with 1 eq the to predict
- Maybe to add a weight 0.5 or something like that in the CoT because if not there are a lot of samples with the similar initial CoT.
- Read again the claude question that is the way that I have to go
- IMPORTANT: I have to create a dataset to validate and check that actually my training is learning what I want in each stage.
- From the 22k take data out to create a validation set -> 1 standard, 1 little endian if possible
- Check how many samples actually generate 6 sols

- Masking randomly the CoT to don't make the llm explicitly learns that and allow divergences.
