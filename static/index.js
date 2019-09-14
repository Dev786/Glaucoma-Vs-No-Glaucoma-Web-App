
$(document).ready(function () {
    let base64Image;
    $("#image-selector").change(function () {
        $("#predicted-data").html("");
        let reader = new FileReader()
        reader.onload = function (e) {
            let dataUrl = reader.result;
            let imageInfo = $("#image-selector")[0].files[0]
            base64Image = dataUrl.replace("data:" + imageInfo.type + ";base64,", "");
            $("#selected-image").attr("src", dataUrl);
        };
        reader.readAsDataURL($("#image-selector")[0].files[0]);
    });
    $("#predict-button").click(function () {
        if (base64Image == undefined) {
            alert("Please Select an Image");
        } else {
            let data = JSON.stringify({ "image": base64Image });
            $.post("/predict", data,
                function (response) {
                    let hasGlaucoma = response.glaucoma == 'Yes';
                    if (hasGlaucoma) {
                        $("#predicted-data").html("Glaucoma has been detected");
                    } else {
                        $("#predicted-data").html("No Glaucoma been detected");
                    }
                }
            );
        }
    });
});
