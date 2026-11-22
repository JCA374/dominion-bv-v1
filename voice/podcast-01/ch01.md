# Episode 01: A challenger that prices every buy

You asked for a new model that could beat our best one. So tonight I started a new project. It asks a simple question. What is each buy actually worth?

Our best model today is the third reinforcement learning network. It wins about eighty six percent of its games against the genetic algorithm base. It plays every decision with one network. When it buys a card, it says what it would buy. It never says how much better that card is than the next one.

The new project tries to measure exactly that. At each buy, a teacher takes the best model's top five choices, plus buying nothing. It plays each one to the very end of the game. Every choice sees the same shuffles of the hidden cards, so the comparison is fair. After that, the best model plays every move for both sides. The share of games won is the price of that buy.

The teacher is slow. One game where it grades every buy takes about half a minute. So the plan is to teach a small, fast network to predict those prices. It looks at the cards in the kingdom as a set, and it scores every buy in one step.

Before any of that, I tested the teacher itself against the best model. It lost. After one hundred games it had about forty one percent. The reason is noise. With only sixteen to thirty two replays per choice, the teacher often thinks a card is better when it was just lucky. It changed the best model's choice on about one buy in five, and many of those changes were mistakes.

The small network has a second chance, because it learns from many positions at once. Luck in one position cancels out across thousands of them. The first small network was trained on only two thousand positions. It learned nothing useful yet. Its prices were flat, and it picked cards it had never been shown. I fixed two things. It now only chooses among the cards the teacher actually graded. And it gets to see the best model's own preferences, so it starts close to them.

Training is running now. It collects twenty thousand graded buys per round, which takes about an hour on this laptop. After each round it plays six hundred test games against the best model. It only overrides a buy when it is confident by a margin.

What I expect is honest uncertainty. Buys differ by a few percent of win chance, and each label is far noisier than that. We probably need a hundred thousand positions or more before the network can see the signal. The goal is fifty three percent against both versions of the best model, on two thousand kingdoms it has never seen. If the dev games stay below fifty percent after a few rounds, I will say so, and we will look at why.

You can follow every round on the project site.
