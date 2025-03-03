$(document).ready(function(){

            $('#input_user').on('input', function(){

                let query = $(this).val();

                if(query.length > 1){
                    $.ajax({
                        url: '/',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({ query: query }),
                        success: function(data){
                            let suggestionsList = '';
                            $.each(data, function(index, value){
                                suggestionsList += '<li>' + value + '</li>';
                            });
                            $('#results').html(suggestionsList);
                        }
                    });
                }
            });

            $(document).on('click', 'li', function(){
                $('#input_user').val($(this).text());
                $('#results').empty();

                var title_choice = $(this).text();
                window.location.href = '/get-movies/' + encodeURIComponent(title_choice);

            });

   });

const input = document.getElementById("input");
let currentIndex = 0;

/**
 * Changing the slide simultaneously while considering the index limit
 * @param value the number of the current slide
 */
function changeIndex(value) {

    const slides = document.getElementsByClassName("slides");
    const slides_size = slides.length;

    currentIndex += value;

    if (currentIndex >= slides_size) {
        currentIndex = 0;
    } else if (currentIndex < 0) {
        currentIndex = slides_size - 1;
    }

    changeSlide(currentIndex);
}

/**
 * Changing a currently visible slide
 * @param y the number of the slide which we want to make visible
 */
function changeSlide(y) {

    let slides = document.getElementsByClassName("slides");
    const slides_size = slides.length;

    for (let i = 0; i < slides_size; i++) {
        slides[i].style.display = "none";
    }

    slides[y].style.display = "flex";
}

/**
 * Changing the slide by clicking the dot
 * @param n the position of the dot in the line
 */
function currentDot(n) {
    currentIndex = n;
    changeSlide(currentIndex);
}

changeSlide(currentIndex);




