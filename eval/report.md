# Retrieval Evaluation

Gold set: 27 questions (22 tune / 5 held-out), page-level relevance, search filtered by bid_id, k=10.

recall@k = share of questions with a relevant page in the top k. Latency p50 is per query on CPU, cache off.

### All questions

| config | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | p50_ms | n |
|---|---|---|---|---|---|---|---|
| vector-only | 0.704 | 0.926 | 0.926 | 0.804 | 0.787 | 11 | 27 |
| bm25-only | 0.778 | 0.963 | 1.0 | 0.865 | 0.877 | 17 | 27 |
| bm25+expand | 0.704 | 1.0 | 1.0 | 0.835 | 0.862 | 15 | 27 |
| hybrid | 0.778 | 0.963 | 0.963 | 0.864 | 0.854 | 17 | 27 |
| hybrid+expand | 0.741 | 0.963 | 0.963 | 0.846 | 0.845 | 17 | 27 |
| hybrid+rerank | 0.852 | 0.963 | 1.0 | 0.9 | 0.888 | 4123 | 27 |
| hybrid+expand+rerank | 0.852 | 0.963 | 1.0 | 0.9 | 0.888 | 3705 | 27 |

### Tune split

| config | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | p50_ms | n |
|---|---|---|---|---|---|---|---|
| vector-only | 0.727 | 0.955 | 0.955 | 0.827 | 0.812 | 11 | 22 |
| bm25-only | 0.818 | 1.0 | 1.0 | 0.898 | 0.901 | 17 | 22 |
| bm25+expand | 0.773 | 1.0 | 1.0 | 0.886 | 0.895 | 15 | 22 |
| hybrid | 0.773 | 1.0 | 1.0 | 0.879 | 0.873 | 17 | 22 |
| hybrid+expand | 0.773 | 1.0 | 1.0 | 0.879 | 0.878 | 17 | 22 |
| hybrid+rerank | 0.864 | 1.0 | 1.0 | 0.917 | 0.902 | 4123 | 22 |
| hybrid+expand+rerank | 0.864 | 1.0 | 1.0 | 0.917 | 0.902 | 3705 | 22 |

### Held-out split (never used for tuning)

| config | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | p50_ms | n |
|---|---|---|---|---|---|---|---|
| vector-only | 0.6 | 0.8 | 0.8 | 0.7 | 0.677 | 11 | 5 |
| bm25-only | 0.6 | 0.8 | 1.0 | 0.72 | 0.768 | 17 | 5 |
| bm25+expand | 0.4 | 1.0 | 1.0 | 0.607 | 0.716 | 15 | 5 |
| hybrid | 0.8 | 0.8 | 0.8 | 0.8 | 0.77 | 17 | 5 |
| hybrid+expand | 0.6 | 0.8 | 0.8 | 0.7 | 0.702 | 17 | 5 |
| hybrid+rerank | 0.8 | 0.8 | 1.0 | 0.825 | 0.827 | 4123 | 5 |
| hybrid+expand+rerank | 0.8 | 0.8 | 1.0 | 0.825 | 0.827 | 3705 | 5 |

### Exact-value questions (bid/part/model numbers)

| config | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | p50_ms | n |
|---|---|---|---|---|---|---|---|
| vector-only | 1.0 | 1.0 | 1.0 | 1.0 | 0.92 | 11 | 6 |
| bm25-only | 0.833 | 1.0 | 1.0 | 0.917 | 0.878 | 17 | 6 |
| bm25+expand | 0.833 | 1.0 | 1.0 | 0.917 | 0.878 | 15 | 6 |
| hybrid | 1.0 | 1.0 | 1.0 | 1.0 | 0.939 | 17 | 6 |
| hybrid+expand | 1.0 | 1.0 | 1.0 | 1.0 | 0.939 | 17 | 6 |
| hybrid+rerank | 0.833 | 1.0 | 1.0 | 0.917 | 0.85 | 4123 | 6 |
| hybrid+expand+rerank | 0.833 | 1.0 | 1.0 | 0.917 | 0.85 | 3705 | 6 |

### Chunk size study (hybrid+expand+rerank)

| config | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | p50_ms | n |
|---|---|---|---|---|---|---|---|
| 1000 chars (~250 tok), 370 chunks | 0.852 | 0.963 | 1.0 | 0.905 | 0.895 | 3809 | 27 |
| 2000 chars (~500 tok), 229 chunks | 0.852 | 0.963 | 1.0 | 0.9 | 0.888 | 3714 | 27 |
| 3000 chars (~750 tok), 180 chunks | 0.852 | 1.0 | 1.0 | 0.902 | 0.883 | 3698 | 27 |

## Questions

| id | bid | split | kind | question | relevant (file p.page) |
|---|---|---|---|---|---|
| q01 | Bid1 | tune | exact | What is the solicitation number of the computing devices RFP? | Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html p.1; JA-207652 Student and Staff Computing Devices FINAL.pdf p.1; JA-207652 Student and Staff Computing Devices FINAL.pdf p.2 |
| q02 | Bid1 | tune | semantic | What is the new proposal due date after the addendum? | Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf p.1 |
| q03 | Bid1 | tune | semantic | When is the pre-proposal meeting and how is it held? | JA-207652 Student and Staff Computing Devices FINAL.pdf p.2; Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html p.1 |
| q04 | Bid1 | heldout | semantic | What minimum warranty is required for student Chromebooks? | Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf p.1; JA-207652 Student and Staff Computing Devices FINAL.pdf p.4 |
| q05 | Bid1 | tune | semantic | Who is the buyer and what is the buyer's email address? | JA-207652 Student and Staff Computing Devices FINAL.pdf p.2 |
| q06 | Bid1 | tune | semantic | Which files must be attached to the submission, such as the W9 form and MWBE forms? | JA-207652 Student and Staff Computing Devices FINAL.pdf p.33 |
| q07 | Bid1 | tune | semantic | What is the length of the contract and how many renewals are allowed? | JA-207652 Student and Staff Computing Devices FINAL.pdf p.2 |
| q08 | Bid1 | tune | semantic | Does Dallas ISD require laser etching on monitors or AIO computers? | Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf p.1 |
| q09 | Bid1 | tune | exact | Is the district a member of a purchasing cooperative such as EPCNT? | JA-207652 Student and Staff Computing Devices FINAL.pdf p.19 |
| q10 | Bid1 | heldout | semantic | When does the district expect to start ordering devices for delivery? | Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf p.1 |
| q11 | Bid1 | tune | semantic | Must the vendor be a manufacturer or authorized reseller? | JA-207652 Student and Staff Computing Devices FINAL.pdf p.3 |
| q12 | Bid1 | tune | exact | Does Form 1295 have to be submitted with the RFP response? | Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf p.4 |
| q13 | Bid1 | tune | semantic | What is the M/WBE subcontracting participation goal percentage? | JA-207652 Student and Staff Computing Devices FINAL.pdf p.50 |
| q14 | Bid1 | tune | semantic | Who should procurement questions about the punch-out catalog setup go to? | JA-207652 Student and Staff Computing Devices FINAL.pdf p.61; JA-207652 Student and Staff Computing Devices FINAL.pdf p.62 |
| q15 | Bid2 | tune | exact | What is the eMMA project number BPM044557 for the Dell laptop bid? | PORFP_-_Dell_Laptop_Final.pdf p.1; Dell Laptops w_Extended Warranty - Bid Information - {3} _ BidNet Direct.html p.1 |
| q16 | Bid2 | tune | semantic | Which affidavit about mercury must the master contractor provide? | PORFP_-_Dell_Laptop_Final.pdf p.2; Mercury_Affidavit.pdf p.1 |
| q17 | Bid2 | heldout | semantic | How many years of extended manufacturer warranty are required? | PORFP_-_Dell_Laptop_Final.pdf p.3 |
| q18 | Bid2 | tune | semantic | How many days after award must the laptops be delivered? | PORFP_-_Dell_Laptop_Final.pdf p.2 |
| q19 | Bid2 | tune | semantic | How should invoices be submitted and within how many days of delivery? | PORFP_-_Dell_Laptop_Final.pdf p.2 |
| q20 | Bid2 | tune | semantic | Who is the agency point of contact and their phone number? | PORFP_-_Dell_Laptop_Final.pdf p.3; Dell Laptops w_Extended Warranty - Bid Information - {3} _ BidNet Direct.html p.1 |
| q21 | Bid2 | heldout | exact | Which item has SKU 210-BLYZ? | Dell_Laptop_Specs.pdf p.1 |
| q22 | Bid2 | tune | semantic | What processor is in the quoted Dell Latitude 5550 configuration? | Dell_Laptop_Specs.pdf p.1 |
| q23 | Bid2 | tune | exact | How many Dell Thunderbolt 4 WD22TB4 docks are requested? | PORFP_-_Dell_Laptop_Final.pdf p.3 |
| q24 | Bid2 | tune | semantic | How are proposals submitted for the Dell laptop PORFP? | PORFP_-_Dell_Laptop_Final.pdf p.1 |
| q25 | Bid2 | tune | semantic | Which master contract must bidders hold to be eligible? | PORFP_-_Dell_Laptop_Final.pdf p.1 |
| q26 | Bid2 | heldout | semantic | Does the contractor certify a drug and alcohol free workplace? | Contract_Affidavit.pdf p.2 |
| q27 | Bid2 | tune | semantic | What is the proposal due date for the Dell laptops? | PORFP_-_Dell_Laptop_Final.pdf p.1; Dell Laptops w_Extended Warranty - Bid Information - {3} _ BidNet Direct.html p.1 |
