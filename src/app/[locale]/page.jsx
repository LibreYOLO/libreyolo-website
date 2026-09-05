import { setRequestLocale } from "next-intl/server";
import LegacyHome from "@/components/home/LegacyHome";
import TaskExplorer from "@/components/home/TaskExplorer";
import { getModelIndex } from "@/lib/models-index";
import { buildPageMetadata, SITE_URL } from "@/i18n/metadata";
export async function generateMetadata({ params }) {
  const { locale } = await params;
  if (locale !== "en") return {};
  return buildPageMetadata({
    title: "LibreYOLO | The MIT-Licensed YOLO Library",
    description:
      "The MIT-licensed YOLO library for training, prediction and export. An alternative to Ultralytics with YOLO9, RF-DETR and a shared Python API.",
    path: "/",
    locale,
  });
}

export default async function Home({ params }) {
  const { locale } = await params;
  setRequestLocale(locale);
  if (locale !== "en") return <LegacyHome />;
  const structuredData = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebSite",
        "@id": `${SITE_URL}/#website`,
        url: `${SITE_URL}/`,
        name: "LibreYOLO",
        inLanguage: "en",
      },
      {
        "@type": "WebPage",
        "@id": `${SITE_URL}/#webpage`,
        url: `${SITE_URL}/`,
        name: "LibreYOLO | The MIT-Licensed YOLO Library",
        description:
          "An MIT-licensed YOLO library and alternative to Ultralytics for training, prediction and export.",
        isPartOf: { "@id": `${SITE_URL}/#website` },
        about: { "@id": `${SITE_URL}/#software` },
        inLanguage: "en",
      },
    ],
  };
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(structuredData).replace(/</g, "\\u003c"),
        }}
      />
      <LegacyHome
        quickstartTabs
        afterSupport={<TaskExplorer tasks={getModelIndex()} />}
      />
    </>
  );
}
