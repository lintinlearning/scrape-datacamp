library(tidyverse)
library(rvest)

course_url <- paste0(
    "https://www.datacamp.com/courses/",
    "introduction-to-statistics-in-python"
)

exercises <- course_url %>%
    read_html() %>%
    html_nodes(xpath = '//div[@class="css-1k6or5q"]//a') %>%
    html_attr("href")

scrape <- function(exercise) {
    sections <- exercise %>%
        read_html() %>%
        html_nodes(xpath = '//div[@class="listview__content"]')

    if (length(sections) == 0) {
        return(NULL)
    }

    section1 <- sections[1] %>%
        as.character()
    section2 <- sections[2] %>%
        html_nodes(xpath = '//div[@class="exercise--instructions__content"]') %>%
        as.character() %>%
        paste(collapse = "")

    instructions_heading <- "<strong>Instructions</strong>"
    solutions_heading <- "<strong>Answer</strong>"

    paste0(section1, instructions_heading, section2, solutions_heading)
}

output <- map(exercises, ~ scrape(.x))

writeLines(unlist(output), file("scrape.html"))
