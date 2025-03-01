#Probing data analysis clean

setwd("~/Library/CloudStorage/OneDrive-Personal/Coding/AISC/SAE_Error_probes")

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

#Truth probe

#Data summaries

truth_df <- read_csv('all_cities.csv')
View(slice_sample(truth_df, n= 10))


probe_name <- "Truth Probe"
some_probes <- read_csv('probe_results_truth.csv')

some_probes <- read_csv('probe_results_hl_frontp.csv')
# Renaming the columns to replace spaces with underscores
colnames(some_probes) <- gsub(" ", "_", colnames(some_probes))

# Modified probe_summary code
probe_summary <- some_probes %>% group_by(`Feature_Type`) %>% 
  summarize(mean_test_accuracy = mean(`Test_Accuracy`), 
            se_test_accuracy = sd(`Test_Accuracy`)/sqrt(n()),
            mean_test_loss = mean(`Test_Loss`), 
            se_test_loss = sd(`Test_Loss`)/sqrt(n())) %>% ungroup()


probe_summary %<>% mutate(graph_labels = c(
  "SAE Error",
  "Residual", 
  "SAE Reconstruction"))


ggplot(probe_summary, aes(x = graph_labels, 
                            y = mean_test_accuracy,)) +
  geom_point() + myTheme + 
  labs(y = 'Probe Out of Sample Accuracy', x = NULL,
       title = str_c(probe_name, " Out of Sample Accuracy"),
       subtitle = "Error bars indicate randomness from using different seeds") + 
  geom_errorbar(aes(ymin = mean_test_accuracy - 1.96*se_test_accuracy, 
                    ymax = mean_test_accuracy + 1.96*se_test_accuracy),
                width = 0.1)+
  scale_y_continuous(labels = scales::percent)


ggplot(probe_summary, aes(x = graph_labels, 
                          y = mean_test_loss,)) +
  geom_point() + myTheme + 
  labs(y = 'Probe Out of Sample Loss', x = NULL,
       title = str_c(probe_name, " Out of Sample Loss"),
       subtitle = "Error bars indicate randomness from using different seeds") + 
  geom_errorbar(aes(ymin = mean_test_loss - 1.96*se_test_loss, 
                    ymax = mean_test_loss + 1.96*se_test_loss),
                width = 0.1)+
  scale_y_continuous()


feols(Test_Loss ~ Feature_Type | as.factor(Seed), data = some_probes, vcov = ~Seed)
feols(Test_Accuracy ~ Feature_Type | as.factor(Seed), data = some_probes, vcov = ~Seed)
