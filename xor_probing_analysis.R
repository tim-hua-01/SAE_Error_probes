#Neel Summer 2025 MATS app

setwd("~/Library/CloudStorage/OneDrive-Personal/Coding/AISC")

if (!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if (!require(magrittr)) install.packages("magrittr"); library(magrittr)
library(estimatr)
library(skimr)
library(fixest)
myTheme <- theme(plot.title = element_text(size = 14),
                 panel.background = element_rect(fill = '#F2F2ED'),
                 axis.title = element_text(size = 10),
                 axis.text = element_text(size = 10, colour = 'black'),
                 legend.title = element_text(size = 12),
                 legend.position = "right",
                 legend.background = element_rect(linetype = 3,size = 0.5, color = 'black', fill = 'grey94'),
                 legend.text = element_text(size = 10),
                 legend.key = element_rect(size = 0.5, linetype = 1, color = 'black'))

#I also have some nice colors that I use in my various graphs.
nicepurp <- "#A88DBF"
niceblue <- '#38A5E0'
nicegreen <- '#A3DCC0'

custom_colors <- c("#2ECC71", "#A3E635", "#F4D03F", "#F39C12", "#E74C3C", "#C0392B")

df <- read_csv('probe_results.csv')

df %>% group_by(Feature_Type, Label) %>% summarise(across(.cols = everything(), .fns = mean))
df %<>% mutate(Feature = case_when(
  Label == "has_alice" ~ "Has Alice",
  Label == "has_alice xor has_not" ~ "Has Alice XOR Has not",
  Label == "has_not" ~ "Has Not",
  Label == "has_not xor label" ~ "Has Not XOR Is True",
  Label == "has_alice xor label" ~ "Has Alice XOR Is True",
  Label == "has_alice xor has_not xor label" ~ "Has Alice XOR Has Not XOR Is True",
  Label == "label" ~ "Is True",
  TRUE ~ Label  # Keeps other values of Label unchanged
),  Input = case_when(Feature_Type == "sae_diff" ~ "SAE Error",
                      Feature_Type == "sae_input" ~ "Residual Stream",
                      TRUE ~ "SAE Recontruction"))

df %>% group_by(Feature, Input) %>% summarise(Mean_test_accuracy = mean(Test_Accuracy)) %>% write_csv("boing.csv")

df %<>% mutate(xor_target = str_detect(Label, 'xor'))

feols(Test_Accuracy ~ as.factor(Feature_Type)*xor_target, data = df, vcov = 'hetero')

accuracy_graph <- df %>% group_by(xor_target,Feature_Type) %>% summarize(mean_accuracy = mean(Test_Accuracy), 
                                                        se_accuracy = sd(Test_Accuracy)/n()) %>% ungroup()

accuracy_graph %<>% mutate(graph_labels = c(
  "Error\n(Basic Feature)",
  "Residual\n(Basic Feature)", 
  "Reconstruction\n(Basic Feature)",
  "Error\n(XOR Feature)", "Residual\n(XOR Feature)",
  "Reconstruction\n(XOR Feature)"))

ggplot(accuracy_graph, aes(x = graph_labels, 
                   y = mean_accuracy, 
                   color = as.factor(xor_target))) +
  geom_point() + myTheme + theme(legend.position = "none") +
  labs(y = 'Probe Out of Sample Accuracy', x = NULL,
       title = "Probing on SAE error yields the most accurate probes, especially for XOR features",
       subtitle = "Error bars indicate randomness from using different seeds") + 
  geom_errorbar(aes(ymin = mean_accuracy - 1.96*se_accuracy, 
                    ymax = mean_accuracy + 1.96*se_accuracy),
                width = 0.1)+
  scale_y_continuous(limits = c(0.90,1), breaks = seq(0,1,0.01), labels = scales::percent)+
  scale_color_manual(values = c(nicepurp, niceblue))
ggsave("probe_accuracy.png", scale = 0.8)

feols(Weight_Norm ~ as.factor(Feature_Type)*xor_target, data = df, vcov = 'hetero')
feols(Cosine_Sim_Input_Diff ~ xor_target, data = df %>% select(Cosine_Sim_Input_Diff,xor_target) %>% distinct(), 
      vcov = 'hetero')


probe_similarities <- read_csv("probe_similarities.csv")

probe_similarities %>%mutate(xor = str_detect(Label, 'xor')) %>% 
  group_by(`Feature Type`) %>% summarise(mean(`Average Cosine Similarity`))



probe_similarities %>%mutate(xor = str_detect(Label, 'xor')) %>% 
  group_by(xor, `Feature Type`) %>% summarise(mean(`Average Cosine Similarity`))
