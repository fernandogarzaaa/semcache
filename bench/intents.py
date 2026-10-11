"""Hand-written paraphrase sets for the load test.

Each key is one *intent* (one correct answer); the list holds different ways
people phrase it. The set is deliberately adversarial: many intents have a
"hard negative" sibling that shares most of its words but needs a different
answer (Python vs. Java, Celsius->Fahrenheit vs. the reverse, cancel vs. pause
a subscription, ...). Serving one for the other is a false hit.

Three domains, the kind of traffic an LLM feature actually caches: a coding
assistant, a general-knowledge helper, and a SaaS customer-support bot.
"""

INTENTS: dict[str, list[str]] = {
    # ---- coding assistant ----------------------------------------------------------------
    "py_reverse_list": [
        "How do I reverse a list in Python?",
        "What's the way to reverse a Python list?",
        "python reverse list",
        "How can I flip the order of a list in Python?",
        "Reverse a list in python, how?",
        "Best way to reverse the elements of a list using Python",
    ],
    "js_reverse_array": [
        "How do I reverse an array in JavaScript?",
        "What's the way to reverse a JS array?",
        "javascript reverse array",
        "How can I flip the order of an array in JavaScript?",
        "Reverse an array in JS, how?",
    ],
    "py_sort_dict_by_value": [
        "How do I sort a dictionary by value in Python?",
        "python sort dict by values",
        "Sort a Python dict by its values",
        "How can I order a dictionary by value in Python?",
        "What's the cleanest way to sort a dict by value in python?",
    ],
    "py_sort_dict_by_key": [
        "How do I sort a dictionary by key in Python?",
        "python sort dict by keys",
        "Sort a Python dict by its keys",
        "How can I order a dictionary by key in Python?",
    ],
    "py_read_file": [
        "How do I read a text file line by line in Python?",
        "python read file line by line",
        "Read each line of a file in Python",
        "What's the way to iterate over lines in a file using Python?",
        "How can I loop through a file's lines in python?",
    ],
    "py_write_file": [
        "How do I write text to a file in Python?",
        "python write string to file",
        "Save a string into a text file with Python",
        "What's the way to write to a file in python?",
    ],
    "py_venv": [
        "How do I create a virtual environment in Python?",
        "python create venv",
        "Set up a Python virtualenv",
        "What's the command to make a new virtual environment for python?",
        "How can I make an isolated python environment with venv?",
    ],
    "git_undo_last_commit": [
        "How do I undo the last git commit?",
        "git undo last commit",
        "Revert my most recent commit in git",
        "How can I take back the last commit I made in git?",
        "What's the way to undo a git commit I just made?",
        "undo previous commit git but keep changes",
    ],
    "git_delete_branch": [
        "How do I delete a local git branch?",
        "git delete local branch",
        "Remove a branch in git locally",
        "What's the command to delete a git branch on my machine?",
    ],
    "git_rename_branch": [
        "How do I rename a git branch?",
        "git rename branch",
        "Change the name of a branch in git",
        "What's the command to rename my current git branch?",
    ],
    "sql_left_vs_inner": [
        "What is the difference between a left join and an inner join?",
        "left join vs inner join",
        "Explain inner join versus left join in SQL",
        "How does a LEFT JOIN differ from an INNER JOIN?",
        "sql inner join compared to left outer join",
    ],
    "sql_count_distinct": [
        "How do I count distinct values in SQL?",
        "sql count unique values",
        "Count the number of distinct rows in a SQL column",
        "What's the query to count unique entries in SQL?",
    ],
    "docker_list_containers": [
        "How do I list running Docker containers?",
        "docker list running containers",
        "Show all containers that are running in docker",
        "What command shows the docker containers currently running?",
    ],
    "docker_remove_images": [
        "How do I remove all unused Docker images?",
        "docker delete unused images",
        "Clean up dangling docker images",
        "What's the command to prune old images in docker?",
    ],
    "regex_email": [
        "What's a regex to validate an email address?",
        "regex for email validation",
        "Regular expression that matches email addresses",
        "Give me a pattern to check if a string is a valid email",
    ],
    "big_o_binary_search": [
        "What is the time complexity of binary search?",
        "binary search big o",
        "How fast is binary search in Big-O terms?",
        "Complexity of binary search algorithm",
    ],
    "big_o_quicksort": [
        "What is the time complexity of quicksort?",
        "quicksort big o",
        "How fast is quicksort in Big-O terms?",
        "Complexity of the quicksort algorithm",
    ],
    "what_is_closure": [
        "What is a closure in programming?",
        "Explain closures",
        "Can you explain what a closure is?",
        "closure meaning in programming",
        "What does it mean when a function is a closure?",
    ],
    "py_list_vs_tuple": [
        "What is the difference between a list and a tuple in Python?",
        "python list vs tuple",
        "Tuple versus list in Python, what's different?",
        "When should I use a tuple instead of a list in python?",
    ],
    "rest_vs_graphql": [
        "What is the difference between REST and GraphQL?",
        "rest vs graphql",
        "Compare GraphQL with REST APIs",
        "How does GraphQL differ from a REST API?",
    ],
    # ---- general knowledge ---------------------------------------------------------------
    "c_to_f": [
        "How do I convert Celsius to Fahrenheit?",
        "celsius to fahrenheit formula",
        "What's the formula to go from C to F?",
        "Convert degrees Celsius into Fahrenheit",
        "How to change celsius into fahrenheit?",
    ],
    "f_to_c": [
        "How do I convert Fahrenheit to Celsius?",
        "fahrenheit to celsius formula",
        "What's the formula to go from F to C?",
        "Convert degrees Fahrenheit into Celsius",
    ],
    "capital_australia": [
        "What is the capital of Australia?",
        "capital city of australia",
        "Which city is Australia's capital?",
        "Australia capital?",
    ],
    "capital_canada": [
        "What is the capital of Canada?",
        "capital city of canada",
        "Which city is Canada's capital?",
        "Canada capital?",
    ],
    "photosynthesis": [
        "How does photosynthesis work?",
        "Explain photosynthesis",
        "What is photosynthesis and how do plants do it?",
        "photosynthesis explained simply",
        "Describe the process of photosynthesis",
    ],
    "why_sky_blue": [
        "Why is the sky blue?",
        "What makes the sky look blue?",
        "why the sky is blue explanation",
        "Explain why the sky appears blue during the day",
        "How come the sky is blue?",
    ],
    "why_sunset_red": [
        "Why are sunsets red?",
        "What makes sunsets look orange and red?",
        "why is the sky red at sunset",
        "Explain why the sky turns red when the sun goes down",
    ],
    "boil_egg": [
        "How long should I boil an egg?",
        "how many minutes to hard boil an egg",
        "Boiling time for a hard-boiled egg",
        "How long do you cook a hard boiled egg?",
    ],
    "km_to_miles": [
        "How many miles are in a kilometer?",
        "convert km to miles",
        "What is one kilometre in miles?",
        "kilometers to miles conversion",
    ],
    "miles_to_km": [
        "How many kilometers are in a mile?",
        "convert miles to km",
        "What is one mile in kilometres?",
        "miles to kilometers conversion",
    ],
    "inflation_meaning": [
        "What is inflation?",
        "Explain inflation in economics",
        "What does inflation mean?",
        "inflation definition simple",
        "Can you explain what inflation is?",
    ],
    "compound_interest": [
        "How does compound interest work?",
        "Explain compound interest",
        "What is compound interest and how is it calculated?",
        "compound interest explained",
    ],
    "simple_interest": [
        "How does simple interest work?",
        "Explain simple interest",
        "What is simple interest and how is it calculated?",
        "simple interest explained",
    ],
    "who_painted_mona_lisa": [
        "Who painted the Mona Lisa?",
        "mona lisa painter",
        "Which artist painted the Mona Lisa?",
        "Who was the painter of the Mona Lisa?",
    ],
    "who_wrote_hamlet": [
        "Who wrote Hamlet?",
        "hamlet author",
        "Which writer wrote the play Hamlet?",
        "Who is the author of Hamlet?",
    ],
    "speed_of_light": [
        "What is the speed of light?",
        "speed of light in m/s",
        "How fast does light travel?",
        "How fast is the speed of light in a vacuum?",
    ],
    "speed_of_sound": [
        "What is the speed of sound?",
        "speed of sound in m/s",
        "How fast does sound travel?",
        "How fast is the speed of sound in air?",
    ],
    "sleep_hours_adult": [
        "How many hours of sleep does an adult need?",
        "recommended sleep for adults",
        "How much sleep should a grown-up get each night?",
        "ideal hours of sleep per night for adults",
    ],
    "water_per_day": [
        "How much water should I drink per day?",
        "daily water intake recommendation",
        "How many glasses of water a day should I drink?",
        "recommended amount of water to drink daily",
    ],
    "ml_vs_dl": [
        "What is the difference between machine learning and deep learning?",
        "machine learning vs deep learning",
        "How is deep learning different from machine learning?",
        "Compare deep learning and machine learning",
    ],
    # ---- SaaS customer support ------------------------------------------------------------
    "reset_password": [
        "How do I reset my password?",
        "I forgot my password, how can I reset it?",
        "reset password",
        "Can't remember my password, what do I do?",
        "How can I change my password if I forgot it?",
        "password reset steps",
    ],
    "change_email": [
        "How do I change the email address on my account?",
        "update account email",
        "I want to use a different email for my account",
        "How can I switch the email linked to my profile?",
    ],
    "cancel_subscription": [
        "How do I cancel my subscription?",
        "cancel subscription",
        "I want to cancel my plan",
        "How can I stop my subscription completely?",
        "Steps to cancel my membership",
        "please cancel my account subscription",
    ],
    "pause_subscription": [
        "How do I pause my subscription?",
        "pause subscription",
        "I want to pause my plan for a month",
        "Can I put my subscription on hold temporarily?",
        "Steps to pause my membership",
    ],
    "refund_policy": [
        "What is your refund policy?",
        "Can I get a refund?",
        "refund policy",
        "How do refunds work if I'm not happy?",
        "Do you offer money back?",
    ],
    "upgrade_plan": [
        "How do I upgrade my plan?",
        "upgrade plan",
        "I want to move to a higher tier plan",
        "How can I switch to the Pro plan?",
        "Steps to upgrade my subscription",
    ],
    "downgrade_plan": [
        "How do I downgrade my plan?",
        "downgrade plan",
        "I want to move to a cheaper plan",
        "How can I switch to the Basic plan?",
        "Steps to downgrade my subscription",
    ],
    "update_payment_method": [
        "How do I update my credit card?",
        "change payment method",
        "I need to update the card on file",
        "How can I add a new payment card to my account?",
        "update billing card",
    ],
    "download_invoice": [
        "Where can I download my invoices?",
        "download invoice",
        "How do I get a copy of my receipt?",
        "Where do I find past invoices for my account?",
        "get billing invoice pdf",
    ],
    "delete_account": [
        "How do I delete my account?",
        "delete my account permanently",
        "I want to close my account and remove my data",
        "How can I permanently remove my account?",
    ],
    "enable_2fa": [
        "How do I turn on two-factor authentication?",
        "enable 2fa",
        "Set up two step verification on my account",
        "How can I add 2FA to my login?",
    ],
    "disable_2fa": [
        "How do I turn off two-factor authentication?",
        "disable 2fa",
        "Remove two step verification from my account",
        "How can I switch off 2FA on my login?",
    ],
    "invite_team_member": [
        "How do I invite a teammate to my workspace?",
        "invite team member",
        "Add a colleague to my team account",
        "How can I give a coworker access to our workspace?",
    ],
    "remove_team_member": [
        "How do I remove a teammate from my workspace?",
        "remove team member",
        "Take a colleague off my team account",
        "How can I revoke a coworker's access to our workspace?",
    ],
    "contact_support": [
        "How can I contact customer support?",
        "talk to a human",
        "What's your support email or phone number?",
        "I need to speak with someone from support",
        "contact support",
    ],
    "export_data": [
        "How do I export my data?",
        "export data to csv",
        "Can I download all my data from the app?",
        "How can I get an export of my account data?",
    ],
    "import_data": [
        "How do I import data from a CSV?",
        "import csv",
        "Can I upload my existing data into the app?",
        "How can I bring in data from a spreadsheet?",
    ],
    "api_rate_limits": [
        "What are the API rate limits?",
        "api rate limit",
        "How many API requests can I make per minute?",
        "Is there a limit on API calls?",
    ],
    "free_trial_length": [
        "How long is the free trial?",
        "free trial length",
        "How many days do I get on the trial?",
        "What is the duration of your free trial?",
    ],
    "student_discount": [
        "Do you have a student discount?",
        "student discount",
        "Is there cheaper pricing for students?",
        "Can students get a discount on the plan?",
    ],
}


def stats() -> tuple[int, int]:
    return len(INTENTS), sum(len(v) for v in INTENTS.values())


if __name__ == "__main__":
    n_intents, n_paraphrases = stats()
    print(f"intents={n_intents} paraphrases={n_paraphrases}")
