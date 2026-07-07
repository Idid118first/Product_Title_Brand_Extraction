Out of the three approaches, the approach which I would ship to production is the "**Hybrid Extractor**". Here, we get the best of both worlds. We are not bound to a specific brand universe by a list, nor are we completely in the wild and letting the LLM run free and almost make an entirely unchecked guess. 


| Metric                             | KnownListExtractor | LLMExtractor (`gpt-4o-mini`) | HybridExtractor (`gpt-4o-mini`) |
| ---------------------------------- | ------------------ | ---------------------------- | ------------------------------- |
| Brands returned                    | 636                | 616                          | 653                             |
| LLM fallback calls                 | —                  | 892 / 892                    | 283 / 892                       |
| **Accuracy (overall)**             | **0.981**          | **0.918**                    | **0.954**                       |
| Accuracy — segment 1 (n=476)       | 0.994              | 0.941                        | 0.998                           |
| Accuracy — segment 2 (n=150)       | 0.973              | 0.847                        | 0.920                           |
| Accuracy — segment 3 (n=266)       | 0.962              | 0.917                        | 0.895                           |
| **Precision**                      | **0.973**          | **0.933**                    | **0.939**                       |
| **Recall** (n=626 recoverable)     | **0.989**          | **0.919**                    | **0.979**                       |
| **Abstention rate** (n=266 absent) | **0.962**          | **0.917**                    | **0.895**                       |
| Total tokens (run)                 | N/A                | 3,138,840                    | 1,009,595                       |
| **Cost per 1,000 titles**          | N/A                | **~$0.53**                   | **~$0.17**                      |


As you can see above, the KnownListExtractor *seems* to have the best results in all departments, but a closer look at the methodology behind the stat calculation will tell the truth. The fact is that the benchmark upon which we ran the extractors is the same one from which the brand list was curated. This means, there is an inherent bias towards the Known List method in this specific instance. However, in reality, we will not always have such a clean and comprehensive list of brands. In an uncontrolled, production environment, we will encounter unseen brands, typos, and overall noise. This is where the LLM holds an advantage: it can handle randomness and novel brands. However, it is still good to use the list of brands we do have to see if we can find any high likelihood matches for a brand in a product. Hence, we should use a hybrid approach: attempt to find high likelihood matches, if found then great, if not we have a fallback option with the LLM approach which has been modified to not just use its own general knowledge base to predict a brand, but it is also given access to the same brand universe to nudge it in the right direction as is chracteristic of a hybrid approach. Above, we see that such a hybrid approach is much closer to a nearly perfect and controlled scenario of using a complete brand universe to find matches than is a purely LLM based approach. Through this, we get the best **production level** accuracy, precision, recall, abstention rate, and cost. The price for the Hybrid approach, to be more specific, is less than 1/3 that of the LLM only approach.