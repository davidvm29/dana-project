window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
            const variable = context.hideout.variable;
            const valor = feature.properties[variable];
            let color = "#cccccc";

            // IGD escala azul
            if (variable === "igd_medio" || variable === "igd_max") {
                if (valor < 1) color = "#d4eeff";
                else if (valor < 4) color = "#74c0e3";
                else if (valor < 10) color = "#1f78b4";
                else color = "#08306b";
            }
            // Daños
            else if (variable === "danio_medio") {
                if (valor < 0.5) color = "#fee5d9";
                else if (valor < 1.0) color = "#fcae91";
                else if (valor < 2.0) color = "#fb6a4a";
                else color = "#cb181d";
            } else if (variable === "danio_max") {
                if (valor < 2) color = "#fee5d9";
                else if (valor < 5) color = "#fcae91";
                else if (valor < 10) color = "#fb6a4a";
                else color = "#cb181d";
            }
            // Edificios
            else if (variable === "edificios") {
                if (valor < 50) color = "#fee5d9";
                else if (valor < 150) color = "#fcae91";
                else if (valor < 400) color = "#fb6a4a";
                else color = "#cb181d";
            }
            // Altura agua
            else if (variable === "altura_media_agua" || variable === "altura_max_agua") {
                if (valor < 0.2) color = "#fee5d9";
                else if (valor < 0.6) color = "#fcae91";
                else if (valor < 1.2) color = "#fb6a4a";
                else color = "#cb181d";
            }

            return {
                fillColor: color,
                color: "#111",
                weight: 1,
                fillOpacity: 0.65
            };
        }

    }
});