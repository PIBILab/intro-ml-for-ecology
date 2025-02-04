#' Dr. Eduardo Vinicius da Silva Oliveira
#' PIBILab  - UFS
#' https://github.com/PIBILab/intro-ml-for-ecology
##################################################################
#' Brief introduction to the Natural language processing

# -------------------------------------------------------------------------
# Packages

install.packages("pdftools")  
install.packages("tm")        
install.packages("stringr")   
install.packages("tm") 
install.packages("e1071") 
library(pdftools)
library(tm)
library(stringr)
library(tm)
library(e1071)

# -------------------------------------------------------------------------

###
#' Search for mentions of species functional traits 

# Defining the directory
pdfs <- "C:/Users/sidfg/OneDrive/Documentos/Eduardo Oliveira/R_analises/ML_PIBILab/pdfs"

# List all files in folder
files_pdf <- list.files(pdfs, pattern = "\\.pdf$", full.names = TRUE)

# Defining the species and functional traits
spp <- "Bertholletia excelsa"  
att <- c("wood density", "seed size", "lifespan")

# Processing the pdf files
resu <- data.frame(
  files = character(),
  spp = character(),
  att = character(),
  stringsAsFactors = FALSE
)

for (files in files_pdf) {
  # Extract text in PDF
  text <- pdf_text(files)
  text_all <- paste(text, collapse = " ")
  
  # Search the functional traits in the text
  att_found <- seek_att(text_all, spp, att)
  
  if (!is.null(att_found)) {
    resu <- rbind(resu, data.frame(
      files = basename(files),
      spp = spp,
      att = paste(att_found, collapse = ", "),
      stringsAsFactors = FALSE
    ))
  }
}

#Check the results
print(resu)

###
#' Search by specific terms 

text2 <- "Dipteryx odorata has high wood density and large seed size."
spp2 <- "Dipteryx odorata"
att2 <- c("wood density", "seed size")

# Verify if species and traits are in the text
if (grepl(spp2, text2, ignore.case = TRUE)) {
  att_found2 <- att2[sapply(att2, function(x) grepl(x, text2, ignore.case = TRUE))]
  print(att2)
}

###
#' Search by specific terms with machine learning
#' Using the model of text classification

text3 <- c("Dipteryx odorata has high wood density.",
            "Bertholletia excelsa is known for its tall stature.")
classes <- c("Dipteryx", "Bertholletia")

# Pre-processing
corpus <- Corpus(VectorSource(text3))
dtm <- DocumentTermMatrix(corpus)

# Train a Naive Bayes model
model <- naiveBayes(as.matrix(dtm), as.factor(classes))

# Predict the class of a new text
new_text <- "This species has high wood density."
new_dtm <- DocumentTermMatrix(Corpus(VectorSource(new_text)), control = list(dictionary = Terms(dtm)))
pred <- predict(model, as.matrix(new_dtm))

#Check the results
print(pred)

rm(list = ls(all.names = TRUE)) #Clear all objects 
gc() #Report the memory usage
# end
######################################################################################################################################
