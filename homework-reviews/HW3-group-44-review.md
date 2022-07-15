# Review of HW3 by Group 44

[Link to the HW](https://1drv.ms/u/s!AtUkm_-Osf4yhZIrxdZrGeSfwoFffg?e=yBesZi)

Dear Group 44,

Here are some thoughts on your assignment #3:

- The implementation seems efficient, and the trained agent performs well (judging from the included videos—I didn't train for the full 4,000 epochs when I ran your code).
- Loss and reward plots that are updated during training are helpful
- The included videos make it easy to instantly evaluate the training success, so this is nicely done
- After every tenth epoch, you show how long the training took during the last epoch. This is nice but perhaps not immediately clear how to interpret this number: if training takes longer, is that good or bad? A short comment that explains the purpose of providing this information would have been nice
- Ample comments
- Reasonable variable names
- In your `train_DQN` function, you could use an additional parameter `min_epsilon` instead of having a hardcoded minimum value below which epsilon does not go

Best wishes,
Group 8