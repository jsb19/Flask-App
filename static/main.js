 jQuery(function () {
    $('.btn').click(function(event) {
        jQuery.ajax({
            url: '/',
            data: {jsdata: event.currentTarget.name},
            type: 'get',
            success: function (response) {
                jQuery("#myChart").attr("src", response)
            }
        });
    });
});


