# Episode 02: The teacher beats our best model

Last time, the teacher lost. It played our best model and won only about forty one percent of the games. It kept changing good buys on the strength of a lucky replay.

So I gave it one rule. It may only change the best model's buy when the gain is clear. The gain has to beat two standard errors of the measurement, and it has to be at least two points. Each choice now gets thirty two replays instead of sixteen.

With that rule the teacher changes only about one buy in twenty-five. And it wins. Against the newer version of our best model it scored fifty six percent over one hundred and sixty games. Against the older version it scored fifty nine percent. In both cases the uncertainty range stays above fifty percent. Those are the first numbers that beat the third reinforcement learning network.

Why does it work? The best model is very good, but it makes a few buys that it would regret if it could see the whole game play out. The teacher plays out each choice to the end, with the best model's own play afterwards, and keeps only the changes that clearly pay.

The catch is speed. The teacher needs about one second for each buy, because it plays a couple of hundred games in its head. For a game on Steam that is fine. For training and big tests it is slow.

That is why the small, fast network is still training. It has now seen about fifty thousand graded buys. It is learning, but slowly. Its picks still lose a little more value than the best model's own picks, so for now it almost never overrides. Its test games sit right at fifty percent. Each buy label is noisy, and it needs many more of them.

Next, a bigger check of the teacher runs on two hundred kingdoms it has never seen, against both versions of the best model. And the small network keeps collecting about twenty thousand graded buys an hour. If it starts to copy the teacher's few good changes, we get the same gain in one fast step.
