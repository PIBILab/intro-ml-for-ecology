# =========================================================
# 'seek_att': function to search functional traits in the text
# This function uses R libraries 'stringr' and 'base'.
# INPUTS: 
#   - 'text': an object of the character type combining all pages in the unique text.
#    - 'spp': an object of the character type with the species name.
#    - 'att': an object of the character type with the traits name. # =========================================================

seek_att <- function(text, spp, att) {
    if (str_detect(tolower(text), tolower(spp))) {
    att_found <- c()
    
    # Search each trait in the text
    for (att in att) {
      if (str_detect(tolower(text), tolower(att))) {
        att_found <- c(att_found, att)
      }
    }
    
    # Returns the found traits
    return(att_found)
  } else {
    return(NULL)  
  }
}

# end of function seek_att
########################################################################################################################################