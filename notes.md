📂 FitLens Backend – File Purpose Notes

1.config.py
Holds important settings like your Supabase link and secret key.
Reads them from the .env file so you don’t hardcode them.

2.db.py
Connects the app to your Supabase database.
Lets other files talk to the database easily.

3.main.py
The main file that runs the app.
Has the routes (API links) for:
Sending a photo + options → getting clothing recommendations.
Tracking likes/clicks from the user.

4.models.py
Lists what kind of data the app expects to receive and send.
Makes sure inputs (like traits, style, size) are in the right format.

5.rules.py
Contains the fashion logic.
Decides what colors, fits, and styles suit the user based on their traits and chosen style (casual/traditional).

6.scoring.py
Scores each product based on:
How well the color matches.
How well the fit matches.
Price compared to budget.
If it’s in stock.
Also gives a short “why we picked this” reason.

7.readme.md
A basic guide for what this project is and how to run it.

8.requirements.txt
List of Python packages the project needs to run.
Used when setting up the project on a new computer.

---
