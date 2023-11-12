// JavaScript Document

var InterfaceMaster = (function () {
    var instance;

    function createInstance() {


        var object = new interfaceObject();

		function interfaceObject(){
			var rss = RSS.getInstance();
			var xml;

			this.init = function(){

			}

			this.displayRSSFeed = function(data){
				xml = data;
				updateXMLDisplay();
			}


			$("#add-post").click(function(e){

				var pubDate = new Date().toUTCString();

				$(xml).find("channel").prepend("<item><title>"+$("#post-title").val()+"</title><description>"+$("#post-content").val()+"</description><link>"+$("#post-link").val()+"</link><pubDate>"+pubDate+"</pubDate></item>");

				updateXMLDisplay();
			});

			function updateXMLDisplay(){
				$("textarea.export").val((new XMLSerializer()).serializeToString(xml));

				var items = rss.feedToObjects(xml);

				$(".feed").html("");

				for(var i = 0; i < items.length; i++){
					var $html = rss.generateItemHTML(items[i]);
					$(".feed").append($html);
				}
			}

			$(".export-xml").click(function(e){
				e.preventDefault();

				var el = $(e.target).prev()[0];
				el.focus();
				el.setSelectionRange(0, el.value.length);
				document.execCommand("copy");
			});

		};

        return object;
    }

    return {
        getInstance: function () {
            if (!instance) {
                instance = createInstance();
            }
            return instance;
        }
    };
})();
