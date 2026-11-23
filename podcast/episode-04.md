# Episode 04: The other half of the turn

You asked a sharp question. Are we only looking at buys? What about playing actions, the order you play them in, and the choice to buy nothing or play nothing? The honest answer was yes. So far we only looked at buys. This episode is about what happened when we looked at the rest.

First, I counted. I played two hundred games of our best model against itself. I recorded every decision with two or more options. A game has about two hundred and forty six of them, counting both players. Buys are only fifty four. That is about one in five. Playing actions is the biggest group, about one hundred and six per game. Then come smaller ones, like what to trash with a Chapel, what to discard to a Militia, and what to gain with an Artisan or a Remodel. Those gains are really buys in disguise, with ten or more options each.

Buying nothing was already covered. The teacher always tests passing as one of its choices. But playing nothing was not. When to stop playing actions was left entirely to the best model.

Second, I looked for the reason the small network stopped improving. I took about four hundred and thirty thousand graded buys. Then I asked a simple question many times. When the best model bought card X, would card Y have been better, on average? I tested seven hundred and forty eight such pairs. Not one of them beat the best model by more than chance. That is a strong result. The best model has no simple buying mistakes, like always taking a Silver where a Market is better.

So where do the teacher's wins come from? From rare exceptions. In a few percent of positions, the exact deck, supply and score make a different buy better. Each graded buy is noisy, much noisier than those small gains. So the small network sees mostly noise, and it has stayed at about fifty percent for many rounds. That is why I paused it.

Third, the experiment. I built an action teacher. It works just like the buy teacher, but at action decisions. When the best model is about to play an action, the teacher tries every legal option, including stopping. It plays each one to the end of the game, thirty two times, on the same shuffles. It only changes the choice when the gain is clearly larger than the noise. Because it is asked again at every action, it also decides the order.

The result surprised me. On actions alone, with the best model doing all the buying, the teacher won fifty six percent over two hundred games. The uncertainty range runs from about fifty one to about sixty one percent. That is as large as the gain from buys.

And it changes very little. It graded about ten thousand action decisions and changed only two hundred and twenty nine. That is about two in a hundred. Most of those changes, one hundred and sixty, played a different action. Sixty nine stopped where the best model would have played on. And not once did it play an action where the best model wanted to stop. So the best model's habit is to play too much, not too little. That fits what you see in Dominion. Playing a terminal action can draw a dead hand or use up your one action, and sometimes it is better to keep it for later.

What does this mean? Our best model was built for one decision at a time, without looking ahead. Looking ahead with full replays fixes a few of its choices in both phases of the turn. Each fix is small, but they add up to several percent of win chance.

Then I ran a combined teacher, grading both buys and actions. It played three hundred games against the best model, and won sixty two percent. The uncertainty range runs from about fifty seven to about sixty six percent. So the two gains really do add up. That is the strongest Dominion player this project has measured.

There is a catch, as always. It is slow, because every graded decision costs a few hundred replays. Next, the plan is to make the teacher faster. And to find a better way to teach a small network these rare exceptions. One idea is to spend many more replays on fewer positions, so that each lesson is clear.
